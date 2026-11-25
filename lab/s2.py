from v5d import *
D0=dict(M=3,B=10,Bl=.5,slack=1,R=99,timed=True)
if __name__=='__main__':
    o2=[(0,1,2,3),(3,2,1,0)]
    with Pool(16) as p:
        g=lambda n=2000,**k: run(p,n,**{**D0,**k})
        for B,Bl in ((10,.5),(15,.5),(10,.6),(20,.4)): g(K=50,J=range(64),B=B,Bl=Bl)
        for K in (46,50,54,58):
            for M in (3,5): g(K=K,J=range(64),M=M,timed=True)
            g(K=K,J=range(64),M=3,orders=o2); g(K=K,J=range(128),M=5)
