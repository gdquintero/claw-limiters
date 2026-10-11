"""Paso 3: comparacion de limitadores en Euler 1D (PyClaw, un paso)."""
import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from euler1d import solve, exact_density_avg
from limlib import minmod, mc, superbee, vanleer, arora_roe, beta_lim, cfl_superbee
from ideas_check import minbee, hybrid
from hyb2 import hyb_d

cf = lambda f: (lambda r, nu: f(r))
H1 = hyb_d(cfl_superbee, 1.0); H05 = hyb_d(cfl_superbee, 0.05); H2 = hyb_d(cfl_superbee, 0.2)
LIMS = {
    "minmod": ("minmod", minmod), "MC": ("MC", mc), "superbee": ("superbee", superbee), "van Leer": ("vanleer", vanleer),
    "MINBEE (idea 1)": ("minbee", cf(minbee)), "Hibrido original (idea 2)": ("hyb0", cf(hybrid)),
    "Arora-Roe": ("AR", arora_roe), "CFL-Superbee": ("CS", cfl_superbee), "Kemm beta=2/3": ("beta", beta_lim),
    "H3 d=1": ("H3d1", H1), "H3 d=0.2": ("H3d.2", H2), "H3 d=0.05": ("H3d.05", H05),
    "H3 acust d=1 / contacto d=0.05": [("H3d1", H1), ("H3d.05", H05), ("H3d1", H1)],
    "H3 d=0.05 / contacto CS": [("H3d.05", H05), ("CS", cfl_superbee), ("H3d.05", H05)],
}
CASES = [("sod", 200), ("lax", 200), ("shuosher", 400), ("blast", 400)]
CFLS = [0.9, 0.5]
NREF = 12800


def reference(prob, N):
    if prob in ("sod", "lax"):
        return exact_density_avg(prob, N)
    _, rf, _, _ = solve(prob, NREF, fortran=True, fortran_lim=4, cfl=0.8)
    return rf.reshape(N, NREF // N).mean(axis=1)


if __name__ == "__main__":
    REF = {p: reference(p, N) for p, N in CASES}
    print("referencias listas", file=sys.stderr)
    R = {}
    for name, lim in LIMS.items():
        for p, N in CASES:
            for c in CFLS:
                try:
                    x, rho, u, pr = solve(p, N, lim, cfl=c)
                    ok = np.all(np.isfinite(rho)) and rho.min() > 0 and pr.min() > 0
                    R[(name, p, c)] = np.abs(rho - REF[p]).mean() * (x[1] - x[0]) * N if ok else np.nan
                except Exception as e:
                    R[(name, p, c)] = np.nan
        print(name, "ok", file=sys.stderr)
    np.save("euler_res.npy", {"R": R, "REF": REF}, allow_pickle=True)
    for c in CFLS:
        print(f"\n=== error L1 de la densidad, CFL {c}  (entre parentesis: error relativo a MC)")
        print(f"{'limitador':34s}" + "".join(f"{p + ' N=' + str(N):>20s}" for p, N in CASES) + "   media rel.")
        for name in LIMS:
            vals = [R[(name, p, c)] for p, N in CASES]
            rel = [v / R[("MC", p, c)] for v, (p, N) in zip(vals, CASES)]
            print(f"{name:34s}" + "".join(f"{v:12.3e} ({r:4.2f})" for v, r in zip(vals, rel)) + f"   {np.exp(np.nanmean(np.log(rel))):.2f}")
