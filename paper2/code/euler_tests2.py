"""Paso 3b: Euler 1D con factor de seguridad theta (Kemm) en ondas no lineales."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from euler1d import solve, exact_density_avg, cell_avg_cons, PROBLEMS
from limlib import mc, superbee, arora_roe, cfl_superbee, beta_lim
from ideas_check import minbee, hybrid, omega
from hyb2 import omega_d
from euler_tests import reference

cf = lambda f: (lambda r, nu: f(r))


def H(delta, theta=1.0):
    def f(r, nu):
        w = omega_d(r, delta)
        return (1 - w) * cfl_superbee(r, nu, theta) + w * arora_roe(r, nu, theta)
    return f


ARt = lambda r, nu: arora_roe(r, nu, 0.75)
L = {
    "MC": ("MC", mc), "superbee": ("superbee", superbee),
    "MINBEE (idea 1)": ("minbee", cf(minbee)), "Hibrido original (idea 2)": ("hyb0", cf(hybrid)),
    "Kemm: AR t=.75 todas": ("ARt", ARt),
    "Kemm: CS contacto + AR t=.75": [("ARt", ARt), ("CS", cfl_superbee), ("ARt", ARt)],
    "N1: H(d=.05,t=.75) todas": ("Hd05t75", H(0.05, 0.75)),
    "N2: H(d=.05) cont + H(d=.05,t=.75)": [("Hd05t75", H(0.05, 0.75)), ("Hd05", H(0.05)), ("Hd05t75", H(0.05, 0.75))],
    "N3: H(d=.05) cont + H(d=1,t=.75)": [("Hd1t75", H(1, 0.75)), ("Hd05", H(0.05)), ("Hd1t75", H(1, 0.75))],
    "N4: H(d=.2) cont + H(d=.05,t=.75)": [("Hd05t75", H(0.05, 0.75)), ("Hd2", H(0.2)), ("Hd05t75", H(0.05, 0.75))],
}
CASES = [("sod", 200), ("lax", 200), ("shuosher", 400), ("blast", 400)]
CFLS = [0.9, 0.5]
NS_ENT = [25, 50, 100, 200, 400, 800]

if __name__ == "__main__":
    REF = {p: reference(p, N) for p, N in CASES}
    R = {}
    for name, lim in L.items():
        for c in CFLS:
            for p, N in CASES:
                try:
                    x, rho, u, pr = solve(p, N, lim, cfl=c)
                    ok = np.all(np.isfinite(rho)) and rho.min() > 0 and pr.min() > 0
                    R[(name, p, c)] = np.abs(rho - REF[p]).mean() * (PROBLEMS[p]["b"] - PROBLEMS[p]["a"]) if ok else np.nan
                except Exception:
                    R[(name, p, c)] = np.nan
            for N in NS_ENT:
                x, rho, u, pr = solve("entropy", N, lim, cfl=c)
                ex = cell_avg_cons(PROBLEMS["entropy"]["ic"], np.linspace(0, 1, N + 1))[0]
                R[(name, "ent", c, N)] = np.abs(rho - ex).mean()
        print(name, "ok", file=sys.stderr)
    np.save("euler_res2.npy", {"R": R, "REF": REF, "L": list(L)}, allow_pickle=True)
    for c in CFLS:
        print(f"\n=== error L1 densidad, CFL {c}   (relativo a MC entre parentesis)")
        print(f"{'esquema':36s}" + "".join(f"{p + str(N):>18s}" for p, N in CASES) + f"{'entropia N=100':>18s}{'N=800':>10s}{'orden':>7s}")
        for name in L:
            vals = [R[(name, p, c)] for p, N in CASES]
            rel = [v / R[("MC", p, c)] for v, (p, N) in zip(vals, CASES)]
            e1, e8 = R[(name, "ent", c, 100)], R[(name, "ent", c, 800)]
            o = np.log2(R[(name, "ent", c, 400)] / e8)
            print(f"{name:36s}" + "".join(f"{v:10.2e} ({r:4.2f})" for v, r in zip(vals, rel)) + f"{e1:18.2e}{e8:10.2e}{o:7.2f}")
