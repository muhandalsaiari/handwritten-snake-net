from v5d import *
D0=dict(M=3,B=10,Bl=.5,slack=1,R=99,timed=True)
if __name__=='__main__':
    with Pool(16) as p:
        for K,n,M,fx in ((46,64,3,1),(46,64,5,1),(46,64,3,2),(48,64,3,1),(48,64,5,1),(48,64,1,1),(48,64,5,2),(46,128,5,1)): run(p,5000,S=2,**{**D0,'K':K,'J':range(n),'M':M,'fix':fx})
