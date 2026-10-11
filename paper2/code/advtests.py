"""One-step wave-propagation (CLAWPACK classic, advection a=1>0) tests of limiters."""
import numpy as np, sys
from limlib import *
from ideas_tests import cell_avg, zalesak

def ratio(up, loc):
    out = np.zeros_like(loc)
    m = loc != 0
    out[m] = up[m] / loc[m]
    return out

def run(q0, nu, nsteps, psi):
    q = q0.copy()
    nuv = np.full_like(q, nu)
    for _ in range(nsteps):
        dq = np.roll(q, -1) - q                      # jump at i+1/2
        r = np.clip(ratio(np.roll(dq, 1), dq), -1e8, 1e8)
        F = q + 0.5 * (1 - nu) * psi(r, nuv) * dq    # CLAWPACK correction: 1/2|s|(1-nu)W~
        q = q - nu * (F - np.roll(F, 1))
    return q

def candidates():
    C = {k: LIMITERS[k] for k in ["MC", "superbee", "AR", "theta-AR(.75)", "CFL-SB", "beta=2/3",
                                  "beta-switch", "Hyperbee", "Superpower-fixed"]}
    C["BL-H1 l=1"] = lambda r, nu: blend(r, nu, 1.0, w_harten)
    C["BL-H1 l=.5"] = lambda r, nu: blend(r, nu, 0.5, w_harten)
    C["BL-H2 l=1"] = lambda r, nu: blend(r, nu, 1.0, lambda r: w_harten(r, 2))
    C["BL-user l=1"] = lambda r, nu: blend(r, nu, 1.0, w_user)
    for e0 in (0.02, 0.05, 0.1, 0.2):
        C[f"RES eta0={e0}"] = (lambda e0: lambda r, nu: resblend(r, nu, e0))(e0)
    return C

if __name__ == "__main__":
    C = candidates()
    nus = [0.1, 0.5, 0.9]
    # T1 Zalesak N=200 t=20
    N = 200; e = np.linspace(-1, 1, N + 1); dx = 2 / N; qz = cell_avg(zalesak, e)
    # T2 sin(6 pi x) T=2 at N=100 and N=800
    # T3 wave packet exp(-50x^2) sin(20 pi x), N=200, T=10
    wp = lambda x: np.exp(-50 * x ** 2) * np.sin(20 * np.pi * x)
    qw = cell_avg(wp, e)
    res = {}
    for name, psi in C.items():
        row = []
        for nu in nus:
            n20 = int(round(20 / (nu * dx)))
            qa = run(qz, nu, n20, psi); ez = np.abs(qa - qz).sum() * dx
            nw = int(round(10 / (nu * dx)))
            qb = run(qw, nu, nw, psi); ew = np.abs(qb - qw).sum() * dx
            es = []
            for Ns in (100, 800):
                es_e = np.linspace(-1, 1, Ns + 1); qs = cell_avg(lambda x: np.sin(6 * np.pi * x), es_e)
                qs2 = run(qs, nu, int(round(2 / (nu * 2 / Ns))), psi); es.append(np.abs(qs2 - qs).sum() * 2 / Ns)
            row.append((ez, ew, es[0], es[1]))
        res[name] = row
        print(name, "done", file=sys.stderr)
    for k, lab in enumerate(["Zalesak L1 (N=200,t=20)", "Wave packet L1 (N=200,t=10)", "sin6pi L1 N=100", "sin6pi L1 N=800"]):
        print(f"\n{lab}")
        print(f"{'limiter':18s}" + "".join(f"  nu={nu:<6}" for nu in nus) + "   mean-rank")
        vals = {n: [res[n][j][k] for j in range(3)] for n in res}
        ranks = {n: 0 for n in res}
        for j in range(3):
            order = sorted(res, key=lambda n: vals[n][j])
            for rk, n in enumerate(order): ranks[n] += (rk + 1) / 3
        for n in sorted(res, key=lambda n: ranks[n]):
            print(f"{n:18s}" + "".join(f"  {v:9.2e}" for v in vals[n]) + f"   {ranks[n]:5.2f}")
