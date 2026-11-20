"""Numeric text and scene-strip composition in the submitted format."""
import re,zlib
import numpy as np
from pdf_objects import read_pdf,write_pdf,stream,replace_stream
NUMERIC=rb'\[\((\d)\)\]TJ/F\d+ .*?\[-17\((\d{3})\)\]'

def number(block):
    m=re.search(NUMERIC,block,re.S);e=re.search(rb'\((\d+)\)\]TJ\s*$',block)
    assert m and e
    sign=-1 if b'[(\\000)]TJ' in block else 1
    return float(m[1]+b'.'+m[2])*10**(sign*int(e[1]))

def patch_number(block,value):
    mantissa,exponent=f'{value:.3e}'.split('e');first,decimals=mantissa.split('.');exponent=int(exponent)
    assert (exponent<0)==(b'[(\\000)]TJ' in block), 'Unexpected exponent-sign change'
    block=re.sub(rb'\[\(\d\)\]TJ',b'[('+first.encode()+b')]TJ',block,count=1)
    block=re.sub(rb'\[-17\(\d{3}\)\]',b'[-17('+decimals.encode()+b')]',block,count=1)
    block=re.sub(rb'\(\d+\)(\]TJ\s*)$',b'('+str(abs(exponent)).encode()+b')'+rb'\1',block,count=1)
    assert abs(number(block)-float(f'{value:.3e}'))<max(1e-14,value*1e-12)
    return block

def remap(body,mapping):
    def refs(s):return re.sub(rb'(?<!\d)(\d+)\s+0\s+R\b',lambda m:f'{mapping[int(m[1])]} 0 R'.encode(),s)
    m=re.search(rb'stream\r?\n',body)
    return refs(body[:m.start()])+body[m.start():] if m else refs(body)

def compose(paths,target,gap):
    objects={};forms=[];next_id=1;sizes=[]
    for path in paths:
        src,root=read_pdf(path);mapping={k:i+next_id for i,k in enumerate(sorted(src))};next_id+=len(src)
        objects.update({mapping[k]:remap(v,mapping) for k,v in src.items()})
        page=next(v for v in src.values() if re.search(rb'/Type\s*/Page\b',v[:100]));media=list(map(float,re.search(rb'/MediaBox\s*\[([^]]+)\]',page)[1].split()));width,height=media[2]-media[0],media[3]-media[1];sizes.append([width,height]);content_id=int(re.search(rb'/Contents\s+(\d+) 0 R',page)[1]);resource_id=int(re.search(rb'/Resources\s+(\d+) 0 R',page)[1])
        form=next_id;next_id+=1;data=stream(src[content_id])[1];raw=zlib.compress(data,9)
        objects[form]=f'<< /Type /XObject /Subtype /Form /BBox [0 0 {width:g} {height:g}] /Resources {mapping[resource_id]} 0 R /Length {len(raw)} /Filter /FlateDecode >>\nstream\n'.encode()+raw+b'\nendstream';forms.append(form)
    assert all(s==sizes[0] for s in sizes);w,h=sizes[0];height=len(paths)*h+(len(paths)-1)*gap
    content=b''.join(f'q 1 0 0 1 0 {(len(paths)-1-i)*(h+gap):.8f} cm /S{i} Do Q\n'.encode() for i in range(len(paths)))
    cid=next_id;page_id=cid+1;pages_id=cid+2;root=cid+3
    objects[cid]=f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream'
    objects[page_id]=f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {w:g} {height:.8f}] /Resources << /XObject << '.encode()+b' '.join(f'/S{i} {form} 0 R'.encode() for i,form in enumerate(forms))+f' >> >> /Contents {cid} 0 R >>'.encode()
    objects[pages_id]=f'<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>'.encode();objects[root]=f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode();write_pdf(target,objects,root)
    return dict(width_pdf_points=w,height_pdf_points=height,scene_strip_sizes=sizes,gap_pdf_points=gap)

