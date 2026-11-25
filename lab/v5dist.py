import v5
from v5 import *
import v5_rule as R
v5.decide=lambda b,f,h,o:R.next_cell(b,f)
if __name__=='__main__':
    with Pool(16) as p: w=[x[1] for x in p.map(game,[(i,{'S':21}) for i in range(4000)],chunksize=8)]
    print('v5 mean',np.mean(w),'share within 1200',np.mean(np.array(w)<=1200))
