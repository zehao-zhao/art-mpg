"""Small PDF object reader/writer for the unencrypted submitted scene strips."""
import re,zlib
from pathlib import Path

def stream(body):
    m=re.search(rb'stream\r?\n',body)
    if not m:return None
    header=body[:m.start()];length=int(re.search(rb'/Length\s+(\d+)\b',header)[1]);raw=body[m.end():m.end()+length]
    return header,zlib.decompress(raw) if b'FlateDecode' in header else raw

def replace_stream(body,data):
    header=stream(body)[0];raw=zlib.compress(data,9) if b'FlateDecode' in header else data
    header=re.sub(rb'/Length\s+\d+\b',b'/Length '+str(len(raw)).encode(),header,count=1)
    return header+b'stream\n'+raw+b'\nendstream'

def read_pdf(path):
    raw=Path(path).read_bytes();objects={}
    for m in re.finditer(rb'(?:\n|^)(\d+) 0 obj\s*\n',raw):
        end=raw.find(b'\nendobj',m.end());objects[int(m[1])]=raw[m.end():end]
    expanded={}
    for body in objects.values():
        ss=stream(body)
        if ss and re.search(rb'/Type\s*/ObjStm',ss[0]):
            header,data=ss;first=int(re.search(rb'/First\s+(\d+)',header)[1]);numbers=list(map(int,data[:first].split()))
            for i in range(0,len(numbers),2):
                stop=numbers[i+3] if i+3<len(numbers) else len(data)-first
                expanded[numbers[i]]=data[first+numbers[i+1]:first+stop]
    objects.update(expanded)
    objects={k:v for k,v in objects.items() if not re.search(rb'/Type\s*/(?:ObjStm|XRef)\b',v[:250])}
    roots=[k for k,v in objects.items() if re.search(rb'/Type\s*/Catalog\b',v[:100])];assert len(roots)==1
    return objects,roots[0]

def write_pdf(path,objects,root):
    data=bytearray(b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n');offsets={}
    for number,body in sorted(objects.items()):
        offsets[number]=len(data);data.extend(f'{number} 0 obj\n'.encode()+body.rstrip()+b'\nendobj\n')
    xref=len(data);size=max(objects)+1;data.extend(f'xref\n0 {size}\n0000000000 65535 f \n'.encode())
    for i in range(1,size):data.extend((f'{offsets[i]:010d} 00000 n \n' if i in offsets else '0000000000 00000 f \n').encode())
    data.extend(f'trailer\n<< /Size {size} /Root {root} 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode());Path(path).write_bytes(data)
