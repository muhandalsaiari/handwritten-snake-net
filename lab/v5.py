"""Fast plain-Python lab for v5 rules (20 cores)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from multiprocessing import Pool
import numpy as np
import build_weights as bw
C=100
NB=[[bw.neighbour(x,m) for m in range(4)] for x in range(C)]
NL=[[y for y in NB[x] if y is not None] for x in range(C)]
DI=[[int(bw.dist(x,y)) for y in range(C)] for x in range(C)]
LN=[next(y for y in NL[x] if DI[x][y]==1) for x in range(C)]
FAR=999
def flood(s,a,timed=False):
    d=[FAR]*C; d[s]=0; q=[s]
    for x in q:
        k=d[x]+1; lim=k if timed else 1
        for y in NL[x]:
            if d[y]==FAR and a[y]<=lim: d[y]=k; q.append(y)
    return d
def tail_dist(a,head,L):
    if L<=2: return 0
    return flood(head,a)[a.index(1)]
def ordered(b):
    return sum(DI[b[i+1]][b[i]] for i in range(len(b)-1))<C
def v3m(h,t,f,L):
    ah=DI[h][t]; best=1; p=LN[h]
    for n in NL[h]:
        j=DI[h][n]
        if j>best and j<=DI[h][f] and j<ah and 2*ah-L-2*(j+5)>0: best=j;p=n
    return p
def trace(d,food,order):
    p=[food]
    while d[p[-1]]>1:
        x=p[-1]
        for m in order:
            y=NB[x][m]
            if y is not None and d[y]==d[x]-1: p.append(y);break
    p.reverse(); return p
def decide(body,food,hun,o):
    L=len(body);h=body[0];a=[0]*C
    for i,c in enumerate(body): a[c]=L-i
    if L>=o['K'] or (o['H'] and hun>o['H']):
        if ordered(body): return v3m(h,body[-1],food,L)
        if all(a[x]<=DI[h][x]-1 for x in body[1:]): return LN[h]
    d=flood(h,a,o['timed'])
    if d[food]<FAR:
        for od in o['orders']:
            p=trace(d,food,od);k=len(p)
            b=[v-k+1 if v>=k else 0 for v in a]
            for i,c in enumerate(p,1): b[c]=max(L+1-k+i,0)
            if tail_dist(b,food,L+1)<FAR: return p[0]
    best=None
    for n in NL[h]:
        if a[n]>=2: continue
        e=1 if n==food else 0
        b=[v-1+e if v else 0 for v in a]; b[n]=L+e
        t=tail_dist(b,n,L+e)
        if t<FAR and (best is None or t>best[0]): best=(t,n)
    if best: return best[1]
    if a[LN[h]]<=1: return LN[h]
    for n in NL[h]:
        if a[n]<=1: return n
    return LN[h]
def game(arg):
    seed,o=arg; r=np.random.default_rng([o.get('S',1),seed]); body=[int(r.integers(C))]
    def nf():
        s=set(body);fr=[c for c in range(C) if c not in s]
        return fr[int(r.integers(len(fr)))] if fr else None
    food=nf();hun=0;bk=[0]*5
    for st in range(1,6001):
        bk[(len(body)-1)//20]+=1
        n=decide(body,food,hun,o)
        if n==food: body.insert(0,n)
        else: body.pop();body.insert(0,n)
        if n in body[1:]: return 'crash',st,bk,len(body)
        hun+=1
        if n==food:
            hun=0;food=nf()
            if food is None: return 'win',st,bk,100
    return 'timeout',st,bk,len(body)
DEF=dict(K=20,H=0,timed=False,orders=[(0,1,2,3)])
def run(pool,G,S=1,**kw):
    o={**DEF,**kw,'S':S};t=time.time()
    R=pool.map(game,[(i,o) for i in range(G)],chunksize=2)
    w=[x for x in R if x[0]=='win'];e={}
    for x in R:e[x[0]]=e.get(x[0],0)+1
    print({k:v for k,v in kw.items()},e,'moves',round(np.mean([x[1] for x in w])) if w else None,'by length',np.mean([x[2] for x in w],axis=0).round().astype(int).tolist() if w else '', [x[3] for x in R if x[0]!='win'][:8],f'{time.time()-t:.0f}s',flush=True)
if __name__=='__main__':
    with Pool(16) as p:
        for K in (20,40,60,80,101): run(p,160,K=K)
        for K in (40,60,80): run(p,160,K=K,timed=True)
        o2=[(0,1,2,3),(3,2,1,0)]
        for K in (40,60,80): run(p,160,K=K,timed=True,orders=o2)
        for K,H in ((40,100),(60,100),(60,200),(80,100),(101,100)): run(p,160,K=K,H=H,timed=True,orders=o2)
