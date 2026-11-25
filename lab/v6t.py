"""Reshaping loop: keep-right on 2x2 blocks; body must stay extendable to a tree loop."""
import v5
from v5 import *
from v6cell import MVS,BL
BE={}      # block pair -> index
EX=[None]*C  # cell -> block-edge decided by that cell's move (None at the wall)
for c in range(C):
    o=MVS[c][1]
    if o is not None: EX[c]=BE.setdefault(tuple(sorted((BL[c],BL[o]))),len(BE))
BP=list(BE)
BN=[[] for _ in range(25)]
for (u,w),e in BE.items(): BN[u].append((w,e));BN[w].append((u,e))
assert len(BE)==40
def ext(body):
    """body[0] head. Can some spanning-tree loop contain this path?"""
    v=[None]*40;bl=set()
    for i in range(len(body)-1,0,-1):
        c=body[i];n=body[i-1];bl.add(BL[c])
        x=1 if n==MVS[c][1] else 0
        if not x and n!=MVS[c][0]: return False
        e=EX[c]
        if e is not None:
            if v[e] is not None and v[e]!=x: return False
            v[e]=x
    bl.add(BL[body[0]])
    if sum(1 for x in v if x)!=len(bl)-1: return False
    seen={BL[body[0]]};q=[BL[body[0]]]
    for u in q:
        for w,e in BN[u]:
            if v[e]!=0 and w not in seen: seen.add(w);q.append(w)
    return len(seen)==25
def tflood(h,a,start=None):
    d=[FAR]*C;d[h]=0;q=[h]
    if start is not None: d[start]=1;q=[start]
    pr=[None]*C
    for x in q:
        k=d[x]+1
        for y in MVS[x]:
            if y is not None and d[y]==FAR and a[y]<=k: d[y]=k;pr[y]=x;q.append(y)
    return d
PRED=[[] for _ in range(C)]
for c in range(C):
    for y in MVS[c]:
        if y is not None: PRED[y].append(c)
def ttrace(d,f,rev=0):
    p=[f]
    while d[p[-1]]>1:
        x=p[-1];c=[y for y in PRED[x] if d[y]==d[x]-1]
        p.append(c[-1] if rev else c[0])
    return p[::-1]
def after(body,p):
    b=p[::-1]+body
    return b[:len(body)+1]
def allpaths(h,a,f,d,cap=200):
    """every shortest path"""
    out=[]
    def rec(x,acc):
        if len(out)>=cap: return
        if d[x]==1: out.append((acc+[x])[::-1]);return
        for y in PRED[x]:
            if d[y]==d[x]-1: rec(y,acc+[x])
    rec(f,[]);return out
def dec(body,food,hun,o):
    L=len(body);h=body[0];a=[0]*C
    for i,c in enumerate(body):a[c]=L-i
    d=tflood(h,a);best=None
    if d[food]<FAR:
        ps=allpaths(h,a,food,d) if o['all'] else [ttrace(d,food),ttrace(d,food,1)][:o['tr']]
        for p in ps:
            if ext(after(body,p)): return p[0]
    ms=[n for n in MVS[h] if n is not None and a[n]<=1]
    sm=[n for n in ms if ext([n]+body[:L if n==food else L-1])]
    if not sm: return (ms or [v5.NL[h][0]])[0]
    if o['deep']:     # other first moves with a path that is extendable at the end
        c=[]
        for n in sm:
            if n==food: return n
            b2=[n]+body[:-1];a2=[0]*C
            for i,x in enumerate(b2):a2[x]=L-i
            d2=tflood(n,a2)
            if d2[food]<FAR:
                ok=any(ext(after(b2,p)) for p in allpaths(n,a2,food,d2,40))
                c.append((d2[food]+(0 if ok else o['pen']),n))
        if c: return min(c)[1]
    if o['pref']=='in': return sm[0]
    if o['pref']=='out': return sm[-1]
    return min(sm,key=lambda n:tflood(n,[0]*C)[food])
v5.decide=dec
B=dict(all=1,tr=2,deep=0,pen=20,pref='in')
if __name__=='__main__':
    with Pool(16) as p:
        for k in (dict(),dict(pref='out'),dict(pref='near'),dict(deep=1),dict(all=0,tr=2),dict(all=0,tr=1)): run(p,320,S=11,**{**B,**k})
