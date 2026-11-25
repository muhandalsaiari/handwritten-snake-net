
from v6u import *
if __name__=='__main__':
    with Pool(16) as p:
        for fb in ('out','alt','tail'):
            for cap in (18,20,24): rr(p,6000,S=31,conf=1,tr=2,cap=cap,fb=fb)
        rr(p,6000,S=31,conf=1,tr=2,cap=40,fb='out')
