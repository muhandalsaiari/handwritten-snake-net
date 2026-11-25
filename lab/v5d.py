import v5, v5b
from v5b import *
def dec(body,food,hun,o):
    L=len(body)
    if L<o['K']: return base(body,food,hun,{**o,'K':999})
    a=[0]*C;h=body[0];J=o['J'];M=o['M']
    for i,c in enumerate(body): a[c]=L-i
    for j in J:
        if ordt(body,TAB[j]):
            D=TAB[j][h];ah=D[body[-1]];best=1;p=NXT[j][h]
            for n in NL[h]:
                x=D[n]
                if x>best and x<=D[food] and x<ah and ah-x-o['B']-o['Bl']*L>0: best=x;p=n
            return p
    if o['slack']:
        best=None
        for j in J:
            D=TAB[j][h];sl=min(D[x]-a[x] for x in body[1:])
            if sl>=0 and (best is None or sl>best[0]): best=(sl,j)
        if best and best[0]>=M or (best and best[0]>=1 and L>=o['K']+o['R']): return NXT[best[1]][h]
    else:
        for j in J:
            if okt(a,body,TAB[j],M): return NXT[j][h]
    if o.get('fix') and best and best[0]>=0:
        ok=False
        for n in NL[h]:
            if a[n]>=2: continue
            e=int(n==food);b=[v-1+e if v else 0 for v in a];b[n]=L+e
            if tail_dist(b,n,L+e)<FAR: ok=True
        if not ok or o['fix']==2: return NXT[best[1]][h]
    return base(body,food,hun,{**o,'K':999})
v5.decide=dec
if __name__=='__main__':
    D0=dict(M=1,B=5,Bl=.5,slack=1,R=99)
    with Pool(16) as p:
        g=lambda n=1000,**k: run(p,n,**{**D0,**k})
        for n in (16,32,64,128):
            for K in (46,50,54): g(K=K,J=range(n))
        for M in (3,5,8):
            for K in (46,50): g(K=K,J=range(64),M=M)
        for M,R in ((5,5),(8,8)): g(K=46,J=range(64),M=M,R=R)
        for B,Bl in ((0,.5),(5,.4),(0,.4),(10,.3),(5,.6),(10,.5),(0,.6),(-10,.7),(-20,.9)): g(K=50,J=range(64),M=3,B=B,Bl=Bl)
