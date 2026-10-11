import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
from euler1d import solve, cell_avg_cons, PROBLEMS
from euler_tests import reference
from euler_tests2 import H, L as L2, CASES

DC = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
L = {k: L2[k] for k in ["MC", "superbee", "MINBEE (idea 1)", "Hibrido original (idea 2)", "Kemm: AR t=.75 todas", "Kemm: CS contacto + AR t=.75"]}
for d in DC:
    L[f"H dc={d}"] = [("Hd1t75", H(1, 0.75)), (f"Hd{d}", H(d)), ("Hd1t75", H(1, 0.75))]
if __name__ == "__main__":
    REF = {p: reference(p, N) for p, N in CASES}
    R = {}
    for name, lim in L.items():
        for c in [0.9, 0.5]:
            for p, N in CASES:
                x, rho, u, pr = solve(p, N, lim, cfl=c)
                R[(name, p, c)] = np.abs(rho - REF[p]).mean() * (PROBLEMS[p]["b"] - PROBLEMS[p]["a"])
            for N in [100, 200, 400, 800]:
                x, rho, u, pr = solve("entropy", N, lim, cfl=c)
                R[(name, "ent", c, N)] = np.abs(rho - cell_avg_cons(PROBLEMS["entropy"]["ic"], np.linspace(0, 1, N + 1))[0]).mean()
        print(name, "ok", file=sys.stderr)
    np.save("euler_pareto.npy", {"R": R, "L": list(L), "DC": DC}, allow_pickle=True)
    for c in [0.9, 0.5]:
        print(f"\n=== CFL {c}: choques = media geometrica del error relativo a MC (sod, lax, shu-osher, blast); suave = entropia")
        for name in L:
            sh = np.exp(np.mean([np.log(R[(name, p, c)] / R[("MC", p, c)]) for p, N in CASES]))
            print(f"  {name:32s} choques {sh:5.2f}   entropia N=100 {R[(name,'ent',c,100)]:.2e}  N=200 {R[(name,'ent',c,200)]:.2e}  N=800 {R[(name,'ent',c,800)]:.2e}")
