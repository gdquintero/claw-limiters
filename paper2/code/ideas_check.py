"""Verification of idea1 (MINBEE) and idea2 (Hybrid MINBEE-TOPUS).
Convention: r = (phi_C - phi_U)/(phi_D - phi_C) = Delta_up/Delta_down  (= Kemm r = LeVeque theta).
"""
import numpy as np
import sympy as sp

# ---------------- limiters -----------------
def superbee(r): return np.maximum(0, np.maximum(np.minimum(1, 2*r), np.minimum(r, 2)))
def minmod(r):   return np.maximum(0, np.minimum(1, r))
def mc(r):       return np.maximum(0, np.minimum(np.minimum(2*r, (1+r)/2), 2))
def vanleer(r):  return (r + np.abs(r))/(1 + np.abs(r))
def minbee(r):   return np.maximum(0, np.minimum(np.minimum(1.5*r, 0.5*r + 0.5), 1.5))
def theta_mc(r, th): return np.maximum(0, np.minimum(np.minimum(2*th*r, (1+r)/2), 2*th))   # Kemm sec 3.1.1
def topus(r):    return np.where(r > 0, r*(6*r + 2)/(1 + np.abs(r))**3, 0.0)
def omega(r):    return (2*np.abs(r)/(r**2 + 1))**2
def hybrid(r):   w = omega(r); return (1 - w)*minbee(r) + w*topus(r)
def koren3(r):   return np.maximum(0, np.minimum(np.minimum(2*r, (2 + r)/3), 2))  # kappa=1/3 limited (semi-discrete 3rd order)

if __name__ == "__main__":
    r = np.linspace(-5, 50, 2_200_001)
    print("MINBEE == (Superbee+Minmod)/2 :", np.abs(minbee(r) - 0.5*(superbee(r) + minmod(r))).max())
    print("MINBEE == theta-MC (theta=3/4):", np.abs(minbee(r) - theta_mc(r, 0.75)).max())

    # NVD form of MINBEE (idea1 eq. 6) vs limiter (eq. 7)
    x = np.linspace(0, 0.999, 100000); rr = x/(1 - x)
    nvd = np.where(x <= 1/3, 1.75*x, np.where(x <= 2/3, x + 0.25, 0.25*x + 0.75))
    print("MINBEE eq.6 vs eq.7          :", np.abs(nvd - (x + 0.5*minbee(rr)*(1 - x))).max())

    # TOPUS: thesis NV polynomial (alpha=2): phi_f = 2x^4 - 3x^3 + 2x  -> limiter
    X, R = sp.symbols('x r', positive=True)
    Pt = 2*X**4 - 3*X**3 + 2*X
    psi_top = sp.factor(sp.simplify(2*(Pt.subs(X, R/(1 + R)) - R/(1 + R))/(1 - R/(1 + R))))
    print("TOPUS from NV polynomial     :", psi_top, "  Psi(1)=", psi_top.subs(R, 1),
          " Psi'(1)=", sp.diff(psi_top, R).subs(R, 1), " Psi'(0)=", sp.limit(sp.diff(psi_top, R), R, 0))
    print("thesis eq 2.40 as printed (alpha=2) Psi(1) =", sp.Rational(1, 1)*8/16)

    # hybrid properties
    rp = np.linspace(1e-9, 200, 4_000_001)
    for name, f in [("MINBEE", minbee), ("TOPUS", topus), ("HYBRID", hybrid), ("MC", mc)]:
        p = f(rp)
        incl = np.all(p >= np.minimum(1, rp) - 1e-12) and np.all(p <= np.maximum(1, rp) + 1e-12)
        mono = np.all(np.diff(p) >= -1e-12)
        sweby = np.all(p <= np.minimum(2*rp, 2) + 1e-12)
        i_out = np.where(p < np.minimum(1, rp) - 1e-9)[0]
        print(f"{name:7s} TVD(Sweby)={sweby} inclusion(M2)={incl} monotone(M5)={mono} max Psi={p.max():.4f} "
              f"sup Psi/r={np.max(p/rp):.4f}  MOL+FE TVD CFL<= {1/(1+0.5*np.max(p/rp)):.4f}"
              + (f"  below minmod for r>{rp[i_out[0]]:.3f}" if len(i_out) else ""))
    h = 1e-6
    d = lambda f, x0: (f(np.array([x0 + h])) - f(np.array([x0 - h])))[0]/(2*h)
    print("slopes at r=1: MINBEE %.4f TOPUS %.4f HYBRID %.4f  (QUICK=1/4, 3rd-order semi-discrete=1/3, Fromm=1/2)"
          % (d(minbee, 1), d(topus, 1), d(hybrid, 1)))
    for rv in [0.1, 0.25, 0.5, 1, 2, 4, 10, 100]:
        a = np.array([rv])
        print(f"  r={rv:6}: omega={omega(a)[0]:.3f} MINBEE={minbee(a)[0]:.3f} TOPUS={topus(a)[0]:.3f} "
              f"HYBRID={hybrid(a)[0]:.3f} MC={mc(a)[0]:.3f}")
    # 1 - omega identity
    print("1-omega == ((1-r^2)/(1+r^2))^2 :", np.abs(1 - omega(rp) - ((1 - rp**2)/(1 + rp**2))**2).max())
