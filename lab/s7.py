
from v6u import *
if __name__=='__main__':
    with Pool(16) as p:
        for cap in (12,16,20,24):
            for tr in (1,2): rr(p,1600,conf=1,tr=tr,cap=cap,H=9999)
        for H,hm in ((60,0),(100,0),(100,1),(60,1),(200,0)):
            rr(p,1600,conf=1,tr=2,cap=20,H=H,hm=hm); rr(p,1600,force=1,conf=2,cap=24,H=H,hm=hm)
        rr(p,1600,force=1,conf=2,tr=1,cap=20,H=100,hm=0); rr(p,1600,conf=1,tr=1,cap=20,H=100,hm=0)
