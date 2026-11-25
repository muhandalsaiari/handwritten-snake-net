"""v5: several loops to switch into."""
import v5, sys, time, numpy as np
from v5 import *
def rnd_loop(r):
    B=5;E=set();e=lambda a,b:(min(a,b),max(a,b));cl=lambda y,x:y*10+x
    for p in range(B):
        for q in range(B):
            y,x=2*p,2*q;E|={e(cl(y,x),cl(y,x+1)),e(cl(y,x+1),cl(y+1,x+1)),e(cl(y+1,x+1),cl(y+1,x)),e(cl(y+1,x),cl(y,x))}
    seen={(0,0)};fr=[((0,0),(0,1)),((0,0),(1,0))]
    while fr:
        u,w=fr.pop(int(r.integers(len(fr))))
        if w in seen: continue
        seen.add(w);(a,b),(c,d)=sorted([u,w])
        if a==c:
            y,x=2*a,2*b+1;E-={e(cl(y,x),cl(y+1,x)),e(cl(y,x+1),cl(y+1,x+1))};E|={e(cl(y,x),cl(y,x+1)),e(cl(y+1,x),cl(y+1,x+1))}
        else:
            y,x=2*a+1,2*b;E-={e(cl(y,x),cl(y,x+1)),e(cl(y+1,x),cl(y+1,x+1))};E|={e(cl(y,x),cl(y+1,x)),e(cl(y,x+1),cl(y+1,x+1))}
        for dy,dx in((0,1),(1,0),(0,-1),(-1,0)):
            z=(w[0]+dy,w[1]+dx)
            if 0<=z[0]<B and 0<=z[1]<B and z not in seen: fr.append((w,z))
    adj={i:[] for i in range(100)}
    for a,b in E: adj[a].append(b);adj[b].append(a)
    o=[0];pr=None;cu=0
    while True:
        n=adj[cu][0] if adj[cu][0]!=pr else adj[cu][1]
        if n==0 and len(o)>2:break
        o.append(n);pr,cu=cu,n
    assert len(o)==100
    return o if r.integers(2) else o[::-1]
def table(o):
    pos={c:i for i,c in enumerate(o)}
    return [[(pos[y]-pos[x])%100 for y in range(100)] for x in range(100)]
r=np.random.default_rng(7)
LOOPS=[list(bw.LOOP),list(bw.LOOP)[::-1]]+[rnd_loop(r) for _ in range(int(os.environ.get("NL","126")))]
TAB=[table(o) for o in LOOPS]
NXT=[[next(y for y in NL[x] if t[x][y]==1) for x in range(100)] for t in TAB]
def v3t(t,nx,h,tl,f,L,B0):
    D=t[h];ah=D[tl];best=1;p=nx[h]
    for n in NL[h]:
        j=D[n]
        if j>best and j<=D[f] and j<ah and 2*ah-L-2*(j+B0)>0: best=j;p=n
    return p
def ordt(b,t):
    s=0
    for i in range(len(b)-1):
        s+=t[b[i+1]][b[i]]
        if s>=100:return False
    return True
def okt(a,b,t,M):
    D=t[b[0]]
    for x in b[1:]:
        if a[x]>D[x]-M:return False
    return True
base=v5.decide
def dec(body,food,hun,o):
    L=len(body)
    if L>=o['K']:
        for j in o['J']:
            if ordt(body,TAB[j]): return v3t(TAB[j],NXT[j],body[0],body[-1],food,L,o.get('B0',5))
        a=[0]*C
        for i,c in enumerate(body): a[c]=L-i
        for j in o['J']:
            if okt(a,body,TAB[j],1): return NXT[j][body[0]]
        if o.get('look'):
            for n in NL[body[0]]:
                if a[n]>=2: continue
                e=int(n==food);b=[n]+(body if e else body[:-1]);a2=[0]*C
                for i,c in enumerate(b): a2[c]=len(b)-i
                for j in o['J']:
                    if okt(a2,b,TAB[j],1): return n
    return base(body,food,hun,{**o,'K':999})
v5.decide=dec
if __name__=='__main__':
    with Pool(16) as p:
        for n in (2,8,32,128):
            for K in (45,55,65): run(p,320,K=K,J=range(n))
        for K in (45,55,65): run(p,320,K=K,J=range(32),look=1)
        run(p,320,K=55,J=range(128),look=1)
