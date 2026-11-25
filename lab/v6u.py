import v6t
from v6t import *
def vars_(body):
    v=[None]*40
    for i in range(len(body)-1,0,-1):
        e=EX[body[i]]
        if e is not None: v[e]=1 if body[i-1]==MVS[body[i]][1] else 0
    return v
def cflood(h,a,v,vis,start=None):
    """conforming flood: respects decisions already made by the body"""
    d=[FAR]*C;d[h]=0;q=[h]
    if start is not None:d[start]=1;q=[start]
    for x in q:
        k=d[x]+1;e=EX[x]
        for t,y in enumerate(MVS[x]):
            if y is None or d[y]<FAR or a[y]>k: continue
            if e is not None and ((t==1 and (v[e]==0 or (v[e] is None and BL[y] in vis))) or (t==0 and v[e]==1)): continue
            d[y]=k;q.append(y)
    return d
def ctrace(d,f,v,vis,rev):
    p=[f]
    while d[p[-1]]>1:
        x=p[-1];c=[]
        for y in PRED[x]:
            if d[y]!=d[x]-1: continue
            e=EX[y];t=1 if x==MVS[y][1] else 0
            if e is not None and ((t==1 and (v[e]==0 or (v[e] is None and BL[x] in vis))) or (t==0 and v[e]==1)): continue
            c.append(y)
        p.append(c[-1] if rev else c[0])
    return p[::-1]
def dec(body,food,hun,o):
    L=len(body);h=body[0];a=[0]*C
    for i,c in enumerate(body):a[c]=L-i
    ms=[n for n in MVS[h] if n is not None and a[n]<=1]
    starts=[None]+([n for n in ms] if o['force'] else [])
    cand=[]
    v=vars_(body);vis={BL[c] for c in body}
    for s in starts:
        d=tflood(h,a,s)
        if d[food]<FAR: cand+=[ttrace(d,food,r) for r in range(o['tr'])]
    if o['conf']:
        for s in ([None] if o['conf']==1 else [None]+ms):
            if s is not None:
                e=EX[h];t=1 if s==MVS[h][1] else 0
                if e is not None and ((t==1 and (v[e]==0 or (v[e] is None and BL[s] in vis))) or (t==0 and v[e]==1)): continue
            d=cflood(h,a,v,vis,s)
            if d[food]<FAR: cand+=[ctrace(d,food,v,vis,r) for r in range(o['tr'])]
    cand.sort(key=len)
    if hun>o['H'] and o['hm']<2: cand=cand[:0] if o['hm']==0 else [ttrace(tflood(h,a),food,r) for r in range(2)] if tflood(h,a)[food]<FAR else []
    for p in cand:
        if len(p)<=o['cap'] and ext(after(body,p)): return p[0]
    sm=[n for n in ms if ext([n]+body[:L if n==food else L-1])]
    if not sm:
        if not ms: return v5.NL[h][0]
        if o['fb']=='out': return ms[-1]
        if o['fb']=='alt': return ms[-1] if hun%7<4 else ms[0]
        # tail: keep-right move that can still reach the tail, outermost first
        for n in ms[::-1]:
            b=[n]+body[:L if n==food else L-1];a2=[0]*C
            for i,c in enumerate(b): a2[c]=len(b)-i
            if tflood(n,a2)[b[-1]]<FAR: return n
        return ms[-1]
    p=o['hp'] if hun>o['H'] else o['pref']
    if p=='in': return sm[0]
    if p=='out': return sm[-1]
    if p=='near': return min(sm,key=lambda n:ZD[n][food])
    return min(sm,key=lambda n:(tflood(n,a)[food],0))
ZD=[tflood(c,[0]*C) for c in range(C)]
v5.decide=dec
B=dict(tr=2,force=0,conf=0,cap=60,pref='out',H=9999,hm=0,hp='out',fb='out')
def rr(p,n=640,S=11,**k):
    o={**v5.DEF,**B,**k,'S':S};R=p.map(game,[(i,o) for i in range(n)],chunksize=4)
    w=[x for x in R if x[0]=='win'];e={}
    for x in R:e[x[0]]=e.get(x[0],0)+1
    print(k,e,round(float(np.mean([x[1] for x in w])),1),np.mean([x[2] for x in w],axis=0).round().astype(int).tolist(),[x[3] for x in R if x[0]!='win'][:6],flush=True)
if __name__=='__main__':
    with Pool(16) as p:
        for pref in ('out','near','timed'):
            for k in (dict(tr=1),dict(),dict(tr=1,conf=1),dict(conf=1),dict(force=1,tr=1),dict(force=1),dict(force=1,conf=2,tr=1),dict(force=1,conf=2)): rr(p,pref=pref,**k)
        for cap in (16,24,32): rr(p,force=1,conf=2,cap=cap)
