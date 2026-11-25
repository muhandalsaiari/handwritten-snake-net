import v5
from v5 import *
import v6_rule as R
v5.decide=lambda b,f,h,o:R.MV[b[0]][R.choose(b,f)]
def find(i): return i,game((i,{'S':41}))
def why(body,food):
    L=len(body);h=body[0];age=[0]*C
    for i,c in enumerate(body): age[c]=L-i
    for nm,ok in (('P',R.FREE),('C',R.allowed_moves(body))):
        d=R.flood(h,age,ok)
        if d[food]<FAR:
            for last in (0,1):
                p=R.trace(d,food,ok,last)
                if R.extendable((p[::-1]+body)[:L+1]): return f'{nm}{last}:{d[food]}'
    for t in (1,0):
        n=R.MV[h][t]
        if n is not None and age[n]<=1 and R.extendable([n]+body[:L if n==food else L-1]): return 'safe '+'IO'[t]
    return 'LAST'
if __name__=='__main__':
    with Pool(16) as p: bad=[i for i,r in p.imap_unordered(find,range(20000),chunksize=50) if r[0]!='win']
    for i in bad[:6]:
        r=np.random.default_rng([41,i]);body=[int(r.integers(C))]
        def nf():
            s=set(body);fr=[c for c in range(C) if c not in s];return fr[int(r.integers(len(fr)))]
        food=nf();hun=0;log=[]
        for st in range(4000):
            if hun>=0: log.append((why(body,food),R.extendable(body)))
            n=v5.decide(body,food,hun,None)
            e=n==food;body=[n]+(body if e else body[:-1]);hun+=1
            if e: hun=0;food=nf();log=[]
            if hun>250: break
        print('game',i,'L',len(body),'first',[(w,'E' if x else 'x') for w,x in log[:45]],'\n  later',{k:log[-120:].count(k) for k in set(log[-120:])})
