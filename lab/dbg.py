from v6u import *
o={**v5.DEF,**B,'S':31,'conf':1,'tr':2,'cap':20,'H':120,'hp':'in','hm':0}
def find(i): return i,game((i,o))
if __name__=='__main__':
    with Pool(16) as p:
        for i,r in p.imap_unordered(find,range(10000),chunksize=20):
            if r[0]!='win': print(i,r[0],r[3]); break
    r=np.random.default_rng([31,i]);body=[int(r.integers(C))]
    def nf():
        s=set(body);fr=[c for c in range(C) if c not in s];return fr[int(r.integers(len(fr)))]
    food=nf();hun=0
    for st in range(3000):
        n=dec(body,food,hun,o)
        if 130<=hun<=150:
            L=len(body);ms=[x for x in MVS[body[0]] if x is not None and x not in body[:-1]]
            print('hun',hun,'head',body[0],'->',n,'food',food,'L',L,'MVS',MVS[body[0]],'legal',ms,'ext',[x for x in ms if ext([x]+body[:L if x==food else L-1])],'self ext',ext(body))
        if hun>150: break
        e=n==food;body=[n]+(body if e else body[:-1]);hun+=1
        if e: hun=0;food=nf()
    print(body)
