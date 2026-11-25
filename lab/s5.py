import v5
from v5 import *
import v5_rule as R
def dec(body,food,hun,o):
    for k,v in o['rk']: setattr(R,k,v)
    return R.next_cell(body,food)
v5.decide=dec
if __name__=='__main__':
    with Pool(16) as p:
        for rk in ([('MARGINS',(8,3))],[('MARGINS',(3,))],[('MARGINS',(8,3)),('DMAX',32)],[('MARGINS',(12,6,3))]): run(p,20000,S=5,rk=rk)
