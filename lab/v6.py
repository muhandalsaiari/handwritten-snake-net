"""v6 lab: shortcuts while straightening, best-loop choice, hunger."""
import v5, v5b
from v5b import *
INV=None
def shortcut_clear(j,body,a,food,S0,S1):
    """best shortcut on loop j that still clears with slack S after moving"""
    h=body[0];L=len(body);D=TAB[j][h];best=1;p=NXT[j][h]
    for n in NL[h]:
        x=D[n]
        if best<x<=D[food] and a[n]<=1:
            e=int(n==food);ok=True;need=S0+S1*L
            Dn=TAB[j][n]
            for c in body:
                v=a[c]-1+e
                if v>0 and c!=n and Dn[c]-v<need: ok=False;break
            if ok: best=x;p=n
    return p
def dec(body,food,hun,o):
    L=len(body)
    K=o['K'] 
    if L<K and not(o['H'] and hun>o['H'] and L>=o.get('KH',20)): return base(body,food,hun,{**o,'K':999})
    a=[0]*C;h=body[0];J=o['J'];M=o['M']
    for i,c in enumerate(body): a[c]=L-i
    od=[]
    for j in J:
        if ordt(body,TAB[j]):
            od.append(j)
            if not o['pick']: break
    if od:
        j=min(od,key=lambda j:TAB[j][h][food]) if o['pick'] and hun<200 else od[0]
        D=TAB[j][h];ah=D[body[-1]];best=1;p=NXT[j][h]
        for n in NL[h]:
            x=D[n]
            if x>best and x<=D[food] and x<ah and ah-x-o['B']-.5*L>0: best=x;p=n
        return p
    cl=[j for j in J if okt(a,body,TAB[j],M)]
    if cl:
        j=min(cl,key=lambda j:TAB[j][h][food]) if o['pick'] and hun<200 else cl[0]
        return shortcut_clear(j,body,a,food,o['S0'],o['S1']) if o['sc'] else NXT[j][h]
    cl=[j for j in J if okt(a,body,TAB[j],0)]
    if cl:
        for n in NL[h]:
            if a[n]>=2: continue
            e=int(n==food);b=[v-1+e if v else 0 for v in a];b[n]=L+e
            if tail_dist(b,n,L+e)<FAR: break
        else: return NXT[cl[0]][h]
    return base(body,food,hun,{**o,'K':999})
v5.decide=dec
B0=dict(K=46,J=range(64),M=3,B=10,H=0,pick=0,sc=0,S0=0,S1=0,timed=True)
def r2(p,n,S=11,**k):
    o={**v5.DEF,**B0,**k}; R=p.map(game,[(i,{**o,'S':S}) for i in range(n)],chunksize=4)
    w=[x for x in R if x[0]=='win'];e={}
    for x in R:e[x[0]]=e.get(x[0],0)+1
    print(k,e,round(float(np.mean([x[1] for x in w])),1),np.mean([x[2] for x in w],axis=0).round().astype(int).tolist(),[x[3] for x in R if x[0]!='win'][:6],flush=True)
if __name__=='__main__':
    with Pool(16) as p:
        g=lambda n=800,**k:r2(p,n,**k)
        g(); g(pick=1)
        for S0,S1 in ((3,0),(8,0),(15,0),(25,0),(0,.3),(5,.4),(10,.5)): g(sc=1,S0=S0,S1=S1)
        for S0,S1 in ((8,0),(15,0),(10,.5)): g(sc=1,S0=S0,S1=S1,pick=1)
        for H in (60,100,200): g(K=54,H=H); 
        g(K=54,sc=1,S0=10,S1=0,pick=1)
        g(K=40,sc=1,S0=10,S1=0,pick=1); g(K=30,sc=1,S0=10,S1=0,pick=1)
