import v7,v7b
from v7b import *
def dec(body,food,hun,o):
    L=len(body);h=body[0];age=ages(body)
    fr=sfree(body,age)
    nb=[None if not fr[t] else [R.MV[h][t]]+body[:L if R.MV[h][t]==food else L-1] for t in (0,1)]
    safe=[fr[t] and EXT(nb[t]) for t in (0,1)]
    al=R.allowed_moves(body);best=None;C_=[]
    for conf,ok in enumerate((R.FREE,al)):
        d=R.flood(h,age,ok)
        if d[food]<FAR:
            for l in (0,1): C_.append((conf,R.trace(d,food,ok,l)))
            if o['mid']:   # two more tie-breaks: prefer IN last / OUT last
                for pref in (0,1):
                    p=[food]
                    while d[p[-1]]>1:
                        c=[(y,t) for y,t in R.PRED[p[-1]] if ok[y][t] and d[y]==d[p[-1]]-1]
                        p.append((([y for y,t in c if t==pref] or [c[0][0]]))[0])
                    C_.append((conf,p[::-1]))
        if o['force']:
            for t in (0,1):
                if safe[t]:
                    d=forced(body,food,age,t,ok)
                    if d:
                        for l in (0,1):
                            p=R.trace(d,food,ok,l)
                            if R.MV[h].index(p[0])==t: C_.append((1,p))
    for conf,p in C_:
        t=R.MV[h].index(p[0]);b=(p[::-1]+body)[:L+1]
        if EXT(b) and (safe[t] or not conf):
            s=len(p)*o['w']+o['fw']*future(b,o)/max(1,C-L-1)
            if best is None or s<best[0]-1e-9: best=(s,p[0])
    if best: return best[1]
    sc=o['sc']
    if sc and all(safe):
        v=[]
        for t in (1,0):
            b=nb[t];a2=ages(b)
            if sc==1: v.append((future(b,o),t))
            else:
                z=R.flood(b[0],a2,R.FREE)[food];v.append((min(z,40)+future(b,o)/(C-L)*(.5),t))
        return R.MV[h][min(v)[1]]
    for t in (1,0):
        if safe[t]: return R.MV[h][t]
    return R.MV[h][1] if fr[1] else R.MV[h][0]
def game2(a):
    import v7c as _m; R.DMAX=a[1]["D"]; v5.decide=_m.dec; return game(a)
def rr(p,n=1500,S=52,**k):
    o={'K':0,'fut':1,'w':1,'fm':0,'fD':20,'D':20,'force':0,'sc':0,'mid':0,'fw':1,**k,'S':S}
    R_=p.map(game2,[(i,o) for i in range(n)],chunksize=4)
    w=[x for x in R_ if x[0]=='win'];e={}
    for x in R_:e[x[0]]=e.get(x[0],0)+1
    m=np.array([x[1] for x in w]);print(k,e,round(float(m.mean()),1),'+-',round(float(m.std()/len(m)**.5),1),np.mean([x[2] for x in w],axis=0).round().astype(int).tolist(),[x[3] for x in R_ if x[0]!='win'][:6],flush=True)
if __name__=='__main__':
    with Pool(16) as p:
        rr(p); rr(p,mid=1); rr(p,D=28); rr(p,force=1); rr(p,force=1,D=28,mid=1)
        for fw in (.5,1.5,2.5): rr(p,fw=fw,force=1)
        for s in (1,2): rr(p,sc=s)
        rr(p,fD=12); rr(p,fD=30)
