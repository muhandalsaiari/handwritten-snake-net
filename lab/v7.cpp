// Fast tester for v7 rules.  g++ -O3 -march=native -pthread v7.cpp -o v7
#include <bits/stdc++.h>
#include <thread>
using namespace std;
const int FAR = 999;
int MV[100][2], BL[100], EDGE[100], NLn[100], NL[100][4], PU[40], PW[40];
vector<pair<int,int>> PRED[100];
int LT[64][100][100], LX[64][100];
struct O { int K=12, D=24, fD=20, force=1, nl=64, M=2, conf=1, sc=0; double w=1, fw=1.5; };
struct S { int b[102]; int L; };
typedef bool OK[100][2];
OK FREE_;

void init() {
  for (int c = 0; c < 100; c++) BL[c] = c/20*5 + c%10/2;
  for (int c = 0; c < 100; c++) {
    int y = c/10, x = c%10; bool t = y%2==0, l = x%2==0; int iy, ix, oy, ox;
    if (t&&l) { iy=0; ix=1; oy=-1; ox=0; } else if (t) { iy=1; ix=0; oy=0; ox=1; }
    else if (!l) { iy=0; ix=-1; oy=1; ox=0; } else { iy=-1; ix=0; oy=0; ox=-1; }
    MV[c][0] = (y+iy)*10 + x+ix;
    int ny = y+oy, nx = x+ox;
    MV[c][1] = (ny<0||ny>9||nx<0||nx>9) ? -1 : ny*10+nx;
    int d[4][2] = {{-1,0},{0,1},{1,0},{0,-1}};
    for (auto& q : d) { int a=y+q[0], b=x+q[1]; if (a>=0&&a<10&&b>=0&&b<10) NL[c][NLn[c]++] = a*10+b; }
  }
  for (int c = 0; c < 100; c++) for (int t = 0; t < 2; t++) if (MV[c][t] >= 0) PRED[MV[c][t]].push_back({c,t});
  map<pair<int,int>,int> mp; int be = 0;
  for (int c = 0; c < 100; c++) {
    if (MV[c][1] < 0) { EDGE[c] = -1; continue; }
    auto k = make_pair(min(BL[c],BL[MV[c][1]]), max(BL[c],BL[MV[c][1]]));
    if (!mp.count(k)) { PU[be]=k.first; PW[be]=k.second; mp[k]=be++; }
    EDGE[c] = mp[k];
  }
  ifstream f("loops7.txt");
  for (int j = 0; j < 64; j++) {
    int o[100], p[100];
    for (int i = 0; i < 100; i++) { f >> o[i]; p[o[i]] = i; }
    for (int i = 0; i < 100; i++) LX[j][o[i]] = o[(i+1)%100];
    for (int x = 0; x < 100; x++) for (int y = 0; y < 100; y++) LT[j][x][y] = (p[y]-p[x]+100)%100;
  }
  for (int c = 0; c < 100; c++) { FREE_[c][0] = 1; FREE_[c][1] = MV[c][1] >= 0; }
}
void ages(const S& s, int* a) { fill(a,a+100,0); for (int i = 0; i < s.L; i++) a[s.b[i]] = s.L-i; }
void dec_(const S& s, bool* d1, bool* d0, bool* vis) {
  fill(d1,d1+40,false); fill(d0,d0+40,false); fill(vis,vis+25,false); vis[BL[s.b[0]]] = 1;
  for (int i = 1; i < s.L; i++) {
    int c = s.b[i], e = EDGE[c]; vis[BL[c]] = 1; if (e < 0) continue;
    if (s.b[i-1] == MV[c][1]) d1[e] = 1; else if (s.b[i-1] == MV[c][0]) d0[e] = 1;
  }
}
bool kr(const S& s) {
  for (int i = 1; i < s.L; i++) if (s.b[i-1] != MV[s.b[i]][0] && s.b[i-1] != MV[s.b[i]][1]) return false;
  return true;
}
bool ext(const S& s) {
  if (!kr(s)) return false;
  bool d1[40], d0[40], vis[25]; dec_(s,d1,d0,vis);
  int n1 = 0, nb = 0;
  for (int e = 0; e < 40; e++) { if (d1[e] && d0[e]) return false; n1 += d1[e]; }
  for (int i = 0; i < 25; i++) nb += vis[i];
  if (n1 != nb-1) return false;
  int p[25], cc = 25; iota(p,p+25,0);
  auto find = [&](int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; };
  for (int e = 0; e < 40; e++) if (!d0[e]) { int a = find(PU[e]), b = find(PW[e]); if (a != b) { p[a] = b; cc--; } }
  return cc == 1;
}
void allowed(const S& s, OK ok) {
  bool d1[40], d0[40], vis[25]; dec_(s,d1,d0,vis);
  for (int c = 0; c < 100; c++) {
    int e = EDGE[c];
    if (e < 0) { ok[c][0] = 1; ok[c][1] = 0; continue; }
    ok[c][0] = !d1[e]; ok[c][1] = !d0[e] && (d1[e] || !vis[BL[MV[c][1]]]);
  }
}
void flood(int h, const int* a, const OK ok, int* d, int DM, int start = -1) {
  fill(d,d+100,FAR); int q[100], n = 0; d[h] = 0;
  if (start >= 0) { d[start] = 1; q[n++] = start; } else q[n++] = h;
  for (int i = 0; i < n; i++) {
    int x = q[i], k = d[x]+1; if (k > DM) break;
    for (int t = 0; t < 2; t++) { int y = MV[x][t]; if (y >= 0 && ok[x][t] && d[y] == FAR && a[y] <= k) { d[y] = k; q[n++] = y; } }
  }
}
int trace(const int* d, int f, const OK ok, int last, int* p) {  // from the food back
  int n = 0; p[n++] = f;
  while (d[p[n-1]] > 1) {
    int x = p[n-1], g = -1;
    for (auto& q : PRED[x]) if (ok[q.first][q.second] && d[q.first] == d[x]-1) { g = q.first; if (!last) break; }
    p[n++] = g;
  }
  return n;
}
S after(const S& s, const int* p, int n) {
  S r; r.L = s.L+1; int i = 0;
  for (; i < n && i < r.L; i++) r.b[i] = p[i];
  for (int j = 0; i < r.L; i++, j++) r.b[i] = s.b[j];
  return r;
}
S mv(const S& s, int n, int f) { S r; r.L = s.L+(n==f); r.b[0] = n; for (int i = 1; i < r.L; i++) r.b[i] = s.b[i-1]; return r; }
double fut(const S& s, int fD) {
  int a[100], d[100]; ages(s,a); flood(s.b[0],a,FREE_,d,fD); double t = 0;
  for (int x = 0; x < 100; x++) if (!a[x]) t += min(d[x],fD+1);
  return t/max(1,100-s.L);
}
int flood4(int h, const int* a, int* d, bool timed) {
  int q[100], n = 0; fill(d,d+100,FAR); d[h] = 0; q[n++] = h;
  for (int i = 0; i < n; i++) { int x = q[i]; if (d[x] >= 24) break; for (int j = 0; j < NLn[x]; j++) { int y = NL[x][j];
    if (d[y] == FAR && a[y] <= (timed ? d[x]+1 : 1)) { d[y] = d[x]+1; q[n++] = y; } } }
  return n;
}
int taild(const S& s) { if (s.L <= 2) return 0; int a[100], d[100]; ages(s,a); flood4(s.b[0],a,d,false); return d[s.b[s.L-1]]; }
int freedec(const S& s, int f, const int* a) {
  int d[100]; flood4(s.b[0],a,d,true); if (d[f] == FAR) return -1;
  for (int l = 0; l < 2; l++) {
    int p[100], m = 0; p[m++] = f;
    while (d[p[m-1]] > 1) { int x = p[m-1], g = -1; for (int j = 0; j < NLn[x]; j++) if (d[NL[x][j]] == d[x]-1) { g = NL[x][j]; if (!l) break; } p[m++] = g; }
    if (taild(after(s,p,m)) < FAR) return p[m-1];
  }
  return -1;
}
int krdec(const S& s, int f, const int* a, const O& o) {
  int h = s.b[0]; bool fr[2], safe[2]; S nb[2];
  for (int t = 0; t < 2; t++) { int n = MV[h][t]; fr[t] = n >= 0 && a[n] <= 1; safe[t] = false; if (fr[t]) { nb[t] = mv(s,n,f); safe[t] = ext(nb[t]); } }
  OK al; allowed(s,al); double bs = 1e9; int bp = -1;
  for (int c = 0; c < (o.conf ? 2 : 1); c++) {
    const OK& ok = c ? al : FREE_;
    for (int st = -1; st < (o.force ? 2 : 0); st++) {
      int d[100];
      if (st >= 0) { if (!safe[st] || !ok[h][st]) continue; flood(h,a,ok,d,o.D,MV[h][st]); } else flood(h,a,ok,d,o.D);
      if (d[f] >= FAR) continue;
      for (int l = 0; l < 2; l++) {
        int p[100], n = trace(d,f,ok,l,p), t = p[n-1] == MV[h][1];
        if (st >= 0 && t != st) continue;
        if ((c || st >= 0) && !safe[t]) continue;
        S r = after(s,p,n); if (!ext(r)) continue;
        double sc = n*o.w + (o.fw ? o.fw*fut(r,o.fD) : 0);
        if (sc < bs-1e-9) { bs = sc; bp = p[n-1]; }
      }
    }
  }
  if (bp >= 0) return bp;
  if (o.sc && safe[0] && safe[1]) return fut(nb[1],o.fD) <= fut(nb[0],o.fD) ? MV[h][1] : MV[h][0];
  for (int t = 1; t >= 0; t--) if (safe[t]) return MV[h][t];
  return fr[1] ? MV[h][1] : MV[h][0];
}
int decide(const S& s, int f, const O& o) {
  int a[100]; ages(s,a); int L = s.L, h = s.b[0];
  if (o.K && (L < o.K || !kr(s))) {
    if (L < o.K) { int n = freedec(s,f,a); if (n >= 0) return n; }
    else {
      int d[100]; flood(h,a,FREE_,d,o.D);
      if (d[f] < FAR) for (int l = 0; l < 2; l++) { int p[100], n = trace(d,f,FREE_,l,p); if (ext(after(s,p,n))) return p[n-1]; }
      int bj = -1;
      for (int j = 0; j < o.nl; j++) {
        bool g = true; for (int i = 1; i < L && g; i++) g = a[s.b[i]]+o.M <= LT[j][h][s.b[i]];
        if (g && bj < 0) bj = j;
      }
      if (bj >= 0) return LX[bj][h];
      bool anyt = false;
      for (int j = 0; j < NLn[h]; j++) { int n = NL[h][j]; if (a[n] < 2 && taild(mv(s,n,f)) < FAR) anyt = true; }
      if (!anyt) for (int j = 0; j < o.nl; j++) {
        bool g = true; for (int i = 1; i < L && g; i++) g = a[s.b[i]] <= LT[j][h][s.b[i]];
        if (g) return LX[j][h];
      }
      int n = freedec(s,f,a); if (n >= 0) return n;
    }
    int bt = -1, bn = -1;
    for (int j = 0; j < NLn[h]; j++) { int n = NL[h][j]; if (a[n] >= 2) continue; int t = taild(mv(s,n,f)); if (t < FAR && t > bt) { bt = t; bn = n; } }
    if (bn >= 0) return bn;
    for (int j = 0; j < NLn[h]; j++) if (a[NL[h][j]] <= 1) return NL[h][j];
    return NL[h][0];
  }
  return krdec(s,f,a,o);
}
struct Rn { uint64_t x; uint32_t operator()(uint32_t n) { x += 0x9E3779B97F4A7C15ull; uint64_t z = x;
  z = (z^(z>>30))*0xBF58476D1CE4E5B9ull; z = (z^(z>>27))*0x94D049BB133111EBull; z ^= z>>31; return (z>>32)*(uint64_t)n>>32; } };
int game(uint64_t seed, const O& o, int* bk, int& Le) {
  Rn r{seed*0x1234567ull+99}; S s; s.L = 1; s.b[0] = r(100);
  auto nf = [&]() { bool u[100] = {}; for (int i = 0; i < s.L; i++) u[s.b[i]] = 1; if (s.L >= 100) return -1;
    int k = r(100-s.L); for (int c = 0; c < 100; c++) if (!u[c] && k-- == 0) return c; return -1; };
  int f = nf(), fe = 0, le_ = 0, ne = 0;
  for (int st = 1; st <= 4000; st++) {
    bk[(s.L-1)/20]++; if (s.L >= max(o.K,2)) { if (ext(s)) { if (!fe) fe = s.L; le_ = s.L; } else ne++; } int n = decide(s,f,o); S t = mv(s,n,f); Le = s.L;
    bool good = false; for (int j = 0; j < NLn[s.b[0]]; j++) good |= NL[s.b[0]][j] == n;
    for (int i = 1; i < t.L; i++) if (t.b[i] == n) good = false;
    if (!good) return -st;
    s = t; if (n == f) { f = nf(); if (f < 0) return st; }
  }
  if (getenv("DBG")) printf("circle kr %d ext %d L %d firstExt %d lastExt %d nonextmoves %d;", kr(s), ext(s), s.L, fe, le_, ne);
  return 0;
}
int main(int ac, char** av) {
  init(); O o; int N = 10000; uint64_t seed = 1; string tag;
  for (int i = 1; i < ac; i++) {
    string s = av[i]; auto p = s.find('='); string k = s.substr(0,p); double v = stod(s.substr(p+1));
    if (k == "N") { N = v; continue; } if (k == "S") { seed = v; continue; } tag += s+" ";
    if (k=="K") o.K=v; else if (k=="D") o.D=v; else if (k=="fD") o.fD=v; else if (k=="force") o.force=v; else if (k=="nl") o.nl=v;
    else if (k=="M") o.M=v; else if (k=="conf") o.conf=v; else if (k=="sc") o.sc=v; else if (k=="w") o.w=v; else if (k=="fw") o.fw=v;
  }
  atomic<int> nx{0}; vector<thread> th; mutex mu; double sum = 0, sq = 0, B[5] = {}; int win = 0, cr = 0, to = 0, bst = 9999, wst = 0, u = 0; map<int,int> bad;
  for (int t = 0; t < 18; t++) th.emplace_back([&] { int i; while ((i = nx++) < N) {
    int bk[5] = {}, Le = 0, r = game(seed*1000003+i,o,bk,Le); lock_guard<mutex> g(mu);
    if (r > 0) { win++; sum += r; sq += (double)r*r; bst = min(bst,r); wst = max(wst,r); u += r <= 1200; for (int k = 0; k < 5; k++) B[k] += bk[k]; }
    else { (r < 0 ? cr : to)++; bad[Le]++; } } });
  for (auto& t : th) t.join();
  double m = sum/win; printf("%s| win %d crash %d circle %d | %.1f +-%.1f best %d worst %d <=1200 %.4f | ", tag.c_str(),win,cr,to,m,sqrt((sq/win-m*m)/win),bst,wst,u/(double)N);
  for (int k = 0; k < 5; k++) printf("%.0f ",B[k]/win); int z = 0; for (auto& b : bad) if (z++ < 8) printf("L%d:%d ",b.first,b.second); puts("");
}
