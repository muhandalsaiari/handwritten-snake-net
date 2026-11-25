import v7
from v7 import *
NL=v5.NL
def allp(d,f,ok,cap):
    out=[]
    def rec(x,acc):
        if len(out)>=cap: return
        if d[x]==1: out.append((acc+[x])[::-1]);return
        for y,t in R.PRED[x]:
            if ok[y][t] and d[y]==d[x]-1: rec(y,acc+[x])
    rec(f,[]);return out
def ages(b):
    a=[0]*C
    for i,c in enumerate(b): a[c]=len(b)-i
    return a
def future(b,o):
    a=ages(b);ok=R.FREE if o['fm']==0 else R.allowed_moves(b)
    D=R.DMAX;R.DMAX=o['fD'];d=R.flood(b[0],a,ok);R.DMAX=D
    return sum(min(d[x],o['fD']+1) for x in range(C) if a[x]==0)
def freedec(body,food,age):
    """early free play (v5 style)"""
    L=len(body);h=body[0]
    d=[FAR]*C;d[h]=0;q=[h]
    for x in q:
        for y in NL[x]:
            if d[y]==FAR and age[y]<=d[x]+1: d[y]=d[x]+1;q.append(y)
    if d[food]==FAR: return None
    for od in (NL,[r[::-1] for r in NL]):
        p=[food]
        while d[p[-1]]>1: p.append(next(y for y in od[p[-1]] if d[y]==d[p[-1]]-1))
        p=p[::-1];b=(p[::-1]+body)[:L+1];a=ages(b)
        if v5.tail_dist(a,food,L+1)<FAR: return p[0]
def kr(b): return all(b[i-1] in R.MV[b[i]] for i in range(1,len(b)))
def dec(body,food,hun,o):
    L=len(body);h=body[0];age=ages(body)
    if L<o['K']:
        n=freedec(body,food,age)
        if n is not None: return n
    fr=sfree(body,age)
    krb=kr(body)
    def EX(b): return kr(b) and EXT(b)
    nb=[None if not fr[t] else [R.MV[h][t]]+body[:L if R.MV[h][t]==food else L-1] for t in (0,1)]
    safe=[fr[t] and EX(nb[t]) for t in (0,1)]
    al=R.allowed_moves(body);best=None
    for conf,ok in enumerate((R.FREE,al)):
        d=R.flood(h,age,ok)
        if d[food]<FAR:
            ps=allp(d,food,ok,o['cap']) if o['cap'] else [R.trace(d,food,ok,l) for l in (0,1)]
            for p in ps:
                t=R.MV[h].index(p[0]);b=(p[::-1]+body)[:L+1]
                if EX(b) and (safe[t] or not conf):
                    if not o['fut']: return p[0]
                    s=len(p)*o['w']+future(b,o)/max(1,C-L-1)
                    if best is None or s<best[0]-1e-9: best=(s,p[0])
    if best: return best[1]
    if not krb:
        # straighten out: keep-right moves with the tail reachable
        for t in (1,0):
            if fr[t]:
                b=nb[t]
                if v5.tail_dist(ages(b),b[0],len(b))<FAR: return b[0]
    for t in (1,0):
        if safe[t]: return R.MV[h][t]
    return R.MV[h][1] if fr[1] else (R.MV[h][0] if fr[0] else next((n for n in NL[h] if age[n]<=1),NL[h][0]))
v7.dec=dec
def rr(p,n=1000,S=51,**k):
    o={'K':0,'cap':0,'fut':0,'w':1,'fm':0,'fD':20,'D':20,**k,'S':S}
    R_=p.map(game2,[(i,o) for i in range(n)],chunksize=4)
    w=[x for x in R_ if x[0]=='win'];e={}
    for x in R_:e[x[0]]=e.get(x[0],0)+1
    print(k,e,round(float(np.mean([x[1] for x in w])),1),np.mean([x[2] for x in w],axis=0).round().astype(int).tolist(),[x[3] for x in R_ if x[0]!='win'][:6],flush=True)
def game2(a):
    R.DMAX=a[1]['D']; v5.decide=dec; return game(a)
if __name__=='__main__':
    with Pool(16) as p:
        rr(p,fut=1); rr(p,fut=1,fm=1); rr(p,fut=1,fm=1,fD=40); rr(p,cap=12); rr(p,cap=12,fut=1,fm=1); rr(p,cap=12,fut=1,fm=0); rr(p,cap=60,fut=1,fm=1)
        rr(p,cap=12,fut=1,fm=1,w=.5); rr(p,cap=12,fut=1,fm=1,w=2)
        for K in (6,10,14,18): rr(p,K=K)
