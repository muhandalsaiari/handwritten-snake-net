import v7c,os
from v7c import *
from v5b import rnd_loop,table
def right(o): return all(o[(i+1)%100] in R.MV[o[i]] for i in range(100))
r_=np.random.default_rng(5);LP=[]
while len(LP)<64:
    o=rnd_loop(r_)
    if not right(o): o=o[::-1]
    assert right(o); LP.append(o)
LT=[table(o) for o in LP]
LX=[[None]*100 for _ in LP]
for j,o in enumerate(LP):
    for i in range(100): LX[j][o[i]]=o[(i+1)%100]
main=v7c.dec
def dec(body,food,hun,o):
    L=len(body);K=o['K']
    if K:
        age=ages(body);k_=kr(body)
        if L<K or not (k_ and EXT(body)):
            mid=False
            if L>=K and k_: # maybe on a verified plain path, or not extendable
                mid=True
            if L<K:
                n=freedec(body,food,age)
                if n is not None: return n
            elif not mid or True:
                # on a verified path?
                al=R.FREE;d=R.flood(body[0],age,al)
                if d[food]<FAR:
                    for l in (0,1):
                        p=R.trace(d,food,al,l);b=(p[::-1]+body)[:L+1]
                        if kr(b) and EXT(b): return p[0]
                h=body[0];c=[]
                for j in range(o['nl']):
                    if all(age[x]+o['M']<=LT[j][h][x] for x in body[1:]): c.append((LT[j][h][food],j))
                if c: return LX[min(c)[1]][h]
                n=freedec(body,food,age)
                if n is not None: return n
            # stall: farthest from tail
            best=None
            for n in NL[body[0]]:
                if age[n]>=2: continue
                e=int(n==food);b=[n]+(body if e else body[:-1]);t=v5.tail_dist(ages(b),n,len(b))
                if t<FAR and (best is None or t>best[0]): best=(t,n)
            return best[1] if best else next((n for n in NL[body[0]] if age[n]<=1),NL[body[0]][0])
    return main(body,food,hun,o)
v7c.dec=dec
if __name__=='__main__':
    q=lambda p,**k: rr(p,1000,**{**dict(force=1,D=24,fw=1.5,nl=64,M=2),**k})
    with Pool(16) as p:
        q(p)
        for K in (8,12,16,20): q(p,K=K)
        q(p,K=12,nl=16); q(p,K=12,M=0)
