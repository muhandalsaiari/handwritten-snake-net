"""Keep-right 2x2 block driving: the loop is a tree of blocks, built as it goes."""
import v5
from v5 import *
def nxtcells(c):
    y,x=divmod(c,10);ey,ex=y%2==0,x%2==0
    # clockwise inside a block; the exit continues straight into the next block
    if ey and ex: ins,out=(0,1),(-1,0)
    elif ey: ins,out=(1,0),(0,1)
    elif not ex: ins,out=(0,-1),(1,0)
    else: ins,out=(-1,0),(0,-1)
    f=lambda d:(y+d[0])*10+x+d[1] if 0<=y+d[0]<10 and 0<=x+d[1]<10 else None
    return f(ins),f(out)
# inside: TL->TR->BR->BL->TL (clockwise); tree outline needs the exits on the left hand
MVS=[nxtcells(c) for c in range(C)]
BL=[(c//10//2)*5+(c%10)//2 for c in range(C)]
def opts(body):
    """legal keep-right moves: turn inside the block, or leave to an empty
    block, or back to the block the body came from."""
    h=body[0];i,o=MVS[h];r=[];bl=[BL[c] for c in body]
    s=set(body[:-1])
    cb=BL[h]
    for n in (i,o):
        if n is None or n in s: continue
        if BL[n]==cb: r.append(n);continue
        if BL[n] not in bl[:-1]: r.append(n);continue
        # parent: last time the body was in BL[n], it left directly into cb
        k=next(k for k,b in enumerate(bl) if b==BL[n])
        if bl[k-1]==cb and all(b!=BL[n] for b in bl[:k]) :
            # is cb's own entry edge still on the body at that point?
            first=k-1
            if cb not in bl[k:]: r.append(n)
    return r
def dec(body,food,hun,o):
    # breadth-first over keep-right paths with the body moving
    from collections import deque
    seen={body[0]};q=deque([(tuple(body),None,0)])
    root=opts(body)
    if not root: return v5.NL[body[0]][0]
    while q:
        b,fs,k=q.popleft()
        for n in opts(list(b)):
            if n in seen: continue
            seen.add(n)
            if n==food: return fs if fs is not None else n
            nb=(n,)+b[:-1]
            q.append((nb,fs if fs is not None else n,k+1))
    return root[0] if hun<300 else root[-1]
v5.decide=dec
if __name__=='__main__':
    with Pool(16) as p: run(p,160)
