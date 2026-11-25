import v5, v5b
from v5b import *
def dec(body,food,hun,o):
    L=len(body)
    if L<o['K']: return base(body,food,hun,{**o,'K':999})
    a=[0]*C;h=body[0];J=o['J']
    for i,c in enumerate(body): a[c]=L-i
    K2=o['K2'];jo=None
    for j in J:
        if ordt(body,TAB[j]): jo=j;break
    if L<K2 and hun<o.get('HM',150):
        d=flood(h,a,o['timed'])
        if d[food]<FAR:
            for od in o['orders']:
                p=trace(d,food,od);k=len(p);b=[v-k+1 if v>=k else 0 for v in a]
                for i,c in enumerate(p,1): b[c]=max(L+1-k+i,0)
                if L+1<o.get('KT',0) and tail_dist(b,food,L+1)<FAR: return p[0]
                vb=[food]+[x for x in range(C) if b[x] and x!=food]
                for j in J:
                    if okt(b,vb,TAB[j],o.get('M',1)): return p[0]
    if jo is not None: return v3t(TAB[jo],NXT[jo],h,body[-1],food,L,o.get('B0',5))
    best=None
    for j in J:
        D=TAB[j][h];sl=min(D[x]-a[x] for x in body[1:])
        if sl>=0 and (best is None or sl>best[0]): best=(sl,j)
        if best and not o.get('slack') : break
    if best and (o.get('slack') or best[0]>=0): return NXT[best[1]][h]
    return base(body,food,hun,{**o,'K':999})
v5.decide=dec
if __name__=='__main__':
    import io
    J=range(int(sys.argv[1])) if len(sys.argv)>1 else range(32)
    with Pool(16) as p:
        for K in (10,30,45,55):
            for K2 in (60,70,80,90,101): run(p,320,K=K,K2=K2,J=J,slack=1,timed=True)
        for M in (3,6): run(p,320,K=45,K2=80,J=J,slack=1,timed=True,M=M)
        run(p,320,K=45,K2=80,J=J,slack=1,timed=True,orders=[(0,1,2,3),(3,2,1,0)])
        run(p,320,K=45,K2=80,J=J,slack=1,timed=False)
