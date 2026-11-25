import sys
import v5
from v5 import *
import v7_rule as R
v5.decide=lambda b,f,h,o:R.next_cell(b,f)
if __name__=='__main__':
    with Pool(16) as p:
        R_=p.map(game,[(i,{"S":61}) for i in range(int(sys.argv[1]))],chunksize=16)
    e={}
    for x in R_: e[x[0]]=e.get(x[0],0)+1
    w=np.array([x[1] for x in R_ if x[0]=='win'])
    print(e,'mean',w.mean().round(1),'best',w.min(),'worst',w.max(),'<=1200',(w<=1200).mean(),[x[3] for x in R_ if x[0]!='win'])
