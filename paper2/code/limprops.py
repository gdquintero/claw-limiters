import numpy as np
from limlib import *

def props(f, nu):
    r = np.concatenate([np.linspace(-20, 0, 2001), np.geomspace(1e-6, 1e4, 400001)])
    p = f(r, np.full_like(r, nu))
    pos = r > 0
    UB = ub(r, nu)
    # general TVD condition (Kemm 17): -2/(1-nu) <= p(r)/r - p(R) <= 2/nu for all r,R
    q = np.where(r != 0, p / np.where(r != 0, r, 1), 0)
    tvd = (q.max() - p.min() <= 2 / nu + 1e-9) and (q.min() - p.max() >= -2 / (1 - nu) - 1e-9)
    restricted = (q.max() - p.min() <= 2 + 1e-9) and (q.min() - p.max() >= -2 - 1e-9)
    incl = np.all(p[pos] >= np.minimum(1, r[pos]) - 1e-9) and np.all(p[pos] <= np.maximum(1, r[pos]) + 1e-9)
    mono = np.all(np.diff(p[pos]) >= -1e-9)
    h = 1e-4
    one = np.array([1 - h, 1.0, 1 + h]); v = f(one, np.full(3, nu))
    sl, sr = (v[1] - v[0]) / h, (v[2] - v[1]) / h
    # third-order tangency: |psi(1+e)-phi3(1+e)| / e^2 bounded?
    e = np.array([1e-2, 1e-3]); dev = np.abs(f(1 + e, np.full(2, nu)) - phi3(1 + e, nu))
    dev2 = np.abs(f(1 - e, np.full(2, nu)) - phi3(1 - e, nu))
    ratio = max([b / a if a > 1e-14 else 0.0 for a, b in [(dev[0], dev[1]), (dev2[0], dev2[1])]])
    tang = "exact" if max(dev.max(), dev2.max()) < 1e-13 else ("O(e^2)" if ratio < 0.02 else "O(e)")
    return tvd, restricted, incl, mono, v[1], sl, sr, tang

if __name__ == "__main__":
    L = dict(LIMITERS)
    L["blend H1 lam=1"] = lambda r, nu: blend(r, nu, 1.0, w_harten)
    L["blend H2 lam=1"] = lambda r, nu: blend(r, nu, 1.0, lambda r: w_harten(r, 2))
    L["blend user lam=1"] = lambda r, nu: blend(r, nu, 1.0, w_user)
    for nu in [0.1, 0.5, 0.9]:
        print(f"\n--- nu={nu}  phi3'={(1+nu)/3:.3f}")
        print(f"{'limiter':18s} genTVD restrTVD  M2    M5   psi(1) slopeL slopeR tangency")
        for n, f in L.items():
            t, rs, i, m, v1, sl, sr, tg = props(f, nu)
            print(f"{n:18s} {str(t):6s} {str(rs):7s} {str(i):5s} {str(m):5s} {v1:6.3f} {sl:6.3f} {sr:6.3f} {tg}")
