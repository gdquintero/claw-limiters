import numpy as np, sys, time, warnings
warnings.filterwarnings("ignore")
from euler2d import run2d, prims, cell_avg2, gresho, PROB2

NL = ("H", 1.0, 0.75)
CONF = {
    "MC": "MC", "superbee": "superbee", "MINBEE (idea 1)": "minbee", "Hibrido original (idea 2)": "hyb0",
    "Kemm: AR t=.75 todas": ("AR", 0.75),
    "Kemm: CS lineales + AR t=.75": [("AR", 0.75), ("CS", 1.0), ("CS", 1.0), ("AR", 0.75)],
    "H dc=0.05": [NL, ("H", 0.05, 1.0), ("H", 0.05, 1.0), NL],
    "H dc=0.3": [NL, ("H", 0.3, 1.0), ("H", 0.3, 1.0), NL],
    "H dc=1": [NL, ("H", 1.0, 1.0), ("H", 1.0, 1.0), NL],
}


def err_quadrants(q, ref):
    k = ref.shape[0] // q.shape[1]
    rref = ref.reshape(q.shape[1], k, q.shape[2], k).mean(axis=(1, 3))
    return np.abs(q[0] - rref).mean()


def err_implosion(X, Y, q, rr, dr):
    r = np.sqrt(X ** 2 + Y ** 2); m = r < 0.9
    return np.abs(q[0][m] - np.interp(r[m], rr, dr)).mean()


def err_gresho(X, Y, q):
    N = X.shape[1]
    xe = np.linspace(0, 4, 4 * N + 1); ye = np.linspace(0, 1, N + 1)
    ex = cell_avg2(lambda x, y: gresho(x, y, PROB2["gresho"]["t"]), xe, ye)
    rho, u, v, p = prims(q)
    return np.mean(np.abs(u - ex[1] / ex[0]) + np.abs(v - ex[2] / ex[0]))


if __name__ == "__main__":
    t0 = time.time()
    REFQ = {}
    for name, spec in [("MC", "MC"), ("Kemm", CONF["Kemm: CS lineales + AR t=.75"])]:
        X, Y, q = run2d("quadrants", 800, spec)
        REFQ[name] = q[0]
        np.save(f"quad_ref800_{name}.npy", q[0])
        print(f"referencia 800x800 {name}: {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    rr, dr = np.load("implosion_ref.npy")
    R = {}
    for name, spec in CONF.items():
        for N in [200]:
            try:
                X, Y, q = run2d("quadrants", N, spec)
                ok = np.isfinite(q).all() and prims(q)[3].min() > 0
                R[(name, "quad-refMC", N)] = err_quadrants(q, REFQ["MC"]) if ok else np.nan
                R[(name, "quad-refKemm", N)] = err_quadrants(q, REFQ["Kemm"]) if ok else np.nan
                np.save(f"quad_{name.replace(' ', '_').replace('/', '')}_{N}.npy", q[0])
            except Exception as e:
                R[(name, "quad-refMC", N)] = R[(name, "quad-refKemm", N)] = np.nan
        for N in [100, 200]:
            try:
                X, Y, q = run2d("implosion", N, spec)
                R[(name, "impl", N)] = err_implosion(X, Y, q, rr, dr) if np.isfinite(q).all() else np.nan
            except Exception:
                R[(name, "impl", N)] = np.nan
        for N in [20, 40, 80]:
            try:
                X, Y, q = run2d("gresho", N, spec)
                R[(name, "gresho", N)] = err_gresho(X, Y, q) if np.isfinite(q).all() else np.nan
            except Exception:
                R[(name, "gresho", N)] = np.nan
        print(name, f"ok {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    np.save("euler2d_res.npy", {"R": R, "CONF": list(CONF)}, allow_pickle=True)
    cols = [("quad-refMC", 200), ("quad-refKemm", 200), ("impl", 100), ("impl", 200), ("gresho", 20), ("gresho", 40), ("gresho", 80)]
    print(f"{'esquema':32s}" + "".join(f"{c + str(n):>17s}" for c, n in cols))
    for name in CONF:
        print(f"{name:32s}" + "".join(f"{R[(name, c, n)]:10.3e} ({R[(name, c, n)] / R[('MC', c, n)]:4.2f})" for c, n in cols))
