from v5d import *
if __name__=='__main__':
    with Pool(16) as p:
        for K in (46,48): run(p,20000,S=3,K=K,J=range(64),M=3,B=10,Bl=.5,slack=1,R=99,timed=True,fix=1)
