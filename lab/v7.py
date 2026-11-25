import v5
from v5 import *
import v6_rule as R
EXT=R.extendable
def plen(d): return d
def cands(body,food,age,o):
    h=body[0]
    for conf,ok in enumerate((R.FREE,R.allowed_moves(body))):
        d=R.flood(h,age,ok)
        if d[food]<FAR:
            for last in (0,1): yield conf,R.trace(d,food,ok,last)
def sfree(body,age): return [n is not None and age[n]<=1 for n in R.MV[body[0]]]
def forced(body,food,age,t,ok):
    """shortest path whose first move is t"""
    h=body[0];n=R.MV[h][t]
    if n is None or age[n]>1 or not ok[h][t]: return None
    d=[FAR]*C;d[h]=0;d[n]=1;q=[n]
    for x in q:
        k=d[x]+1
        if k>R.DMAX: break
        for u in (0,1):
            y=R.MV[x][u]
            if y is not None and ok[x][u] and d[y]==FAR and age[y]<=k: d[y]=k;q.append(y)
    return d if d[food]<FAR else None
def dec(body,food,hun,o):
    L=len(body);h=body[0];age=[0]*C
    for i,c in enumerate(body): age[c]=L-i
    fr=sfree(body,age)
    nb=[None if not fr[t] else [R.MV[h][t]]+body[:L if R.MV[h][t]==food else L-1] for t in (0,1)]
    safe=[fr[t] and EXT(nb[t]) for t in (0,1)]
    al=R.allowed_moves(body)
    for conf,ok in enumerate((R.FREE,al)):
        d=R.flood(h,age,ok)
        if d[food]<FAR:
            for last in (0,1):
                p=R.trace(d,food,ok,last);t=R.MV[h].index(p[0])
                if EXT((p[::-1]+body)[:L+1]) and (safe[t] or not conf): return p[0]
    if o['force']:
        best=None
        for t in (1,0):
            if not safe[t]: continue
            for ok in (R.FREE,al):
                d=forced(body,food,age,t,ok)
                if d is None: continue
                for last in (0,1):
                    p=R.trace(d,food,ok,last)
                    if R.MV[h].index(p[0])==t and EXT((p[::-1]+body)[:L+1]) and (best is None or len(p)<best[0]): best=(len(p),t)
        if best: return R.MV[h][best[1]]
    if o['look'] and all(safe):
        # compare the two safe moves by the shortest path from each
        sc=[]
        for t in (1,0):
            b=nb[t];a2=[0]*C
            for i,c in enumerate(b): a2[c]=len(b)-i
            if o['look']==1: v=R.flood(b[0],a2,R.allowed_moves(b))[food]
            elif o['look']==2: v=min(R.flood(b[0],a2,R.FREE)[food],99)
            else:
                v=999
                for ok in (R.FREE,R.allowed_moves(b)):
                    d=R.flood(b[0],a2,ok)
                    if d[food]<FAR:
                        for last in (0,1):
                            p=R.trace(d,food,ok,last)
                            if EXT((p[::-1]+b)[:L+1]): v=min(v,len(p))
            sc.append((v,-t))
        return R.MV[h][-min(sc)[1]]
    for t in (1,0):
        if safe[t]: return R.MV[h][t]
    return R.MV[h][1] if fr[1] else R.MV[h][0]
v5.decide=dec
def rr(p,n=1000,S=51,**k):
    o={'force':0,'look':0,'D':20,**k,'S':S}
    R_=p.map(game2,[(i,o) for i in range(n)],chunksize=4)
    w=[x for x in R_ if x[0]=='win'];e={}
    for x in R_:e[x[0]]=e.get(x[0],0)+1
    print(k,e,round(float(np.mean([x[1] for x in w])),1),np.mean([x[2] for x in w],axis=0).round().astype(int).tolist(),[x[3] for x in R_ if x[0]!='win'][:6],flush=True)
def game2(a):
    R.DMAX=a[1]['D']; v5.decide=dec; return game(a)
if __name__=='__main__':
    with Pool(16) as p:
        rr(p)
        for D in (28,40): rr(p,D=D)
        rr(p,force=1); rr(p,force=1,D=30)
        for l in (1,2,3): rr(p,look=l)
        rr(p,look=1,D=40); rr(p,look=3,D=30); rr(p,force=1,look=3,D=30); rr(p,force=1,look=1,D=40)
