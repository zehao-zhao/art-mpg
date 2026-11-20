#!/usr/bin/env python3
"""Train and export the joint R/T model used in equal-time tables."""
import argparse,json,math,os,random,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT/'models'
sys.path.insert(0,str(WORKSPACE/'src'))
from masked_diffusion_mpg.dataset import ReplayTrainingDataset
from masked_diffusion_mpg.training import split_dataset_indices

class JointRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_layers=nn.ModuleList([nn.Linear(22,40),nn.Linear(40,40),nn.Linear(40,40)])
        self.direction_layers=nn.ModuleList([nn.Linear(50,40),nn.Linear(40,40),nn.Linear(40,40)])
        self.joint_head=nn.Linear(40,1022)
        self.vmf_logits_head=nn.Linear(40,4)
        self.vmf_mean_head=nn.Linear(40,12)
        self.vmf_kappa_head=nn.Linear(40,4)
    def forward(self,x,length,bits):
        h=x
        for layer in self.context_layers:h=F.dropout(F.silu(layer(h)),.1,self.training)
        logits=self.joint_head(h)
        d=torch.cat([h,length.float().unsqueeze(-1)/9,bits.float()],-1)
        for layer in self.direction_layers:d=F.dropout(F.silu(layer(d)),.1,self.training)
        raw=self.vmf_mean_head(d).view(-1,4,3)
        norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
        fallback=torch.zeros_like(raw);fallback[...,2]=1
        means=torch.where(norm>1e-6,raw/norm.clamp_min(1e-6),fallback)
        kappas=(F.softplus(self.vmf_kappa_head(d))+1e-4).clamp_max(64)
        return logits,self.vmf_logits_head(d),means,kappas

def loss(model,x,n,b,direction,weight):
    logits,mix,mu,kappa=model(x,n,b)
    code=sum((b[:,bit].long()<<bit) for bit in range(9))
    target=(1<<n)-2+code
    ce=F.cross_entropy(logits,target,reduction='none')
    log_sinh=kappa+torch.log1p(-torch.exp(-2*kappa))-math.log(2)
    log_c=torch.log(kappa)-math.log(4*math.pi)-log_sinh
    log_pdf=F.log_softmax(mix,-1)+log_c+kappa*(mu*direction[:,None,:]).sum(-1)
    nll=-torch.logsumexp(log_pdf,-1)
    return ((ce+nll)*weight).sum()/weight.sum()

def export(model,path):
    tensors={}
    for prefix,layers in [('cat_context',model.context_layers),('cat_direction',model.direction_layers)]:
        for i,layer in enumerate(layers):
            tensors[f'{prefix}_{i}_weight']=layer.weight.detach()
            tensors[f'{prefix}_{i}_bias']=layer.bias.detach()
    for name in ('joint_head','vmf_logits_head','vmf_mean_head','vmf_kappa_head'):
        layer=getattr(model,name);tensors[name+'_weight']=layer.weight.detach();tensors[name+'_bias']=layer.bias.detach()
    with path.open('x') as f:
        f.write('masked_diffusion_runtime_v1\nhidden_dim 40\ndepth 3\ncondition_dim 22\nbackbone_type joint_categorical\n')
        f.write('direction_distribution vmf_mixture\nvmf_components 4\nvmf_min_kappa 0.0001\nvmf_max_kappa 64\n')
        f.write('runtime_input_mode joint_cat_geometry\ndiscrete_diffusion_mode ordered_absorbing\nchain_length_classes 10\nchain_type_bits 9\n')
        for name,tensor in tensors.items():
            a=tensor.cpu().numpy();f.write('tensor '+name+' '+str(a.ndim)+' '+' '.join(map(str,a.shape))+'\n')
            f.write(' '.join(format(float(v),'.9g') for v in a.reshape(-1))+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    output=a.output;output.mkdir(parents=True,exist_ok=False)
    replay=a.replay
    torch.set_num_threads(1);torch.manual_seed(104729);random.seed(104729);np.random.seed(104729)
    dataset=ReplayTrainingDataset((replay,))
    train,val=split_dataset_indices(len(dataset),val_fraction=.25,seed=104729,group_keys=dataset.group_keys)
    data={key:torch.stack([dataset[i][key] for i in range(len(dataset))]) for key in
          ('condition','chain_length','chain_type_bits','direction','sample_weight')}
    model=JointRT();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best=float('inf');best_state=None;begin=time.perf_counter()
    for epoch in range(32):
        model.train();permutation=torch.randperm(len(train)).tolist();values=[]
        for start in range(0,len(train),16):
            idx=[train[j] for j in permutation[start:start+16]]
            optimizer.zero_grad();value=loss(model,*(data[k][idx] for k in data))
            if not torch.isfinite(value):raise RuntimeError('Nonfinite simple R/T training loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();values.append(float(value.detach()))
        model.eval()
        with torch.no_grad():validation=float(loss(model,*(data[k][val] for k in data))) if val else float(np.mean(values))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(values)),validation_loss=validation))
        print(a.scene,'epoch',epoch+1,'validation',validation,flush=True)
        if validation<best:best=validation;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    torch.save(dict(state_dict=best_state,architecture='joint_categorical_rt_h40_d3_k4',model_seed=104729),output/'model.pt')
    export(model,output/'runtime.txt')
    config=dict(model='joint categorical R/T',parameter_count=sum(p.numel() for p in model.parameters()),model_seed=104729,epochs=32,batch_size=16,lr=.001,weight_decay=.0001,
       dropout=.1,gradient_clip=.5,weight='same replay1+log1p(energy)',loss='weighted joint descriptorCE + 4-vMF directionalNLL',
       replay=str(replay),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),records=len(dataset),train_indices=train,validation_indices=val,
       direction_space='S2, no hemisphere fold',render_seeds=[1729,31415,271828,424242,8675309])
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (output/'training.json').write_text(json.dumps(dict(scene=a.scene,method='art_joint',threads=1,clock_scope='training, validation, checkpoint and runtime export; excludes replay loading',training_seconds=time.perf_counter()-begin,
         best_validation_loss=best,parameter_count=config['parameter_count'],runtime_sha256=hashlib.sha256((output/'runtime.txt').read_bytes()).hexdigest()),indent=2)+'\n')
    print(a.scene,config['parameter_count'],len(dataset),'trained',flush=True)

if __name__=='__main__':main()
