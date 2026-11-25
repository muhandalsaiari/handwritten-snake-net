from v6 import *
if __name__=='__main__':
    n=len(TAB)
    with Pool(16) as p:
        for K in (46,50,54): r2(p,800 if n<600 else 400,S=12,K=K,J=range(n),pick=1)
