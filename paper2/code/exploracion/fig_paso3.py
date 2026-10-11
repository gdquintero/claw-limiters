import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
_EXP = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(_EXP, '..'))  # datos .npy en ../
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from euler1d import solve, cell_avg_cons, PROBLEMS
from euler_pareto import L, DC

D = np.load("euler_pareto.npy", allow_pickle=True).item(); R = D["R"]
CASES = [("sod", 200), ("lax", 200), ("shuosher", 400), ("blast", 400)]
c = 0.9
sh = lambda n: np.exp(np.mean([np.log(R[(n, p, c)] / R[("MC", p, c)]) for p, N in CASES]))
sm = lambda n: R[(n, "ent", c, 200)]
fig, ax = plt.subplots(2, 2, figsize=(13, 10.5)); ax = ax.ravel()
hs = [f"H dc={d}" for d in DC]
ax[0].loglog([sh(n) for n in hs], [sm(n) for n in hs], "-o", color="tab:blue", label="híbrido mejorado, variando δ (contacto)")
for n in hs:
    ax[0].annotate(n.split("=")[1], (sh(n), sm(n)), textcoords="offset points", xytext=(6, -3), fontsize=8, color="tab:blue")
ref = [("MC", "0.3", "s", "MC"), ("superbee", "tab:purple", "v", "Superbee"),
       ("MINBEE (idea 1)", "tab:orange", "P", "MINBEE (tu idea 1)"), ("Hibrido original (idea 2)", "tab:red", "s", "híbrido original (tu idea 2)"),
       ("Kemm: AR t=.75 todas", "tab:olive", "^", "Kemm: Arora–Roe θ=0.75"), ("Kemm: CS contacto + AR t=.75", "tab:green", "D", "Kemm: mejor combinación")]
for n, col, m, lab in ref:
    ax[0].loglog(sh(n), sm(n), m, color=col, ms=9, label=lab)
ax[0].set_xlabel("error en choques/contactos (relativo a MC; Sod, Lax, Shu–Osher, explosión)")
ax[0].set_ylabel("error en flujo suave (onda de entropía, N=200)")
ax[0].set_title("(a) Euler: abajo a la izquierda = mejor (CFL 0.9)"); ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

show = [("Hibrido original (idea 2)", "tab:red", "-", "híbrido original"), ("MC", "0.5", ":", "MC"),
        ("Kemm: CS contacto + AR t=.75", "tab:green", "--", "Kemm: mejor combinación"), ("H dc=0.05", "tab:blue", "-", "híbrido mejorado (δ=0.05)")]
for k, (prob, N, xl, title) in enumerate([("shuosher", 400, (0.4, 2.6), "(b) Shu–Osher, N=400 (zona de ondas de entropía)"),
                                          ("blast", 400, (0.6, 0.86), "(c) Explosión de Woodward–Colella, N=400")]):
    a = ax[1 + k]
    xf, rf, _, _ = solve(prob, 12800, fortran=True, fortran_lim=4, cfl=0.8)
    a.plot(xf, rf, "k", lw=0.8, label="referencia (malla 12800)")
    for n, col, ls, lab in show:
        x, r, _, _ = solve(prob, N, L[n], cfl=c)
        a.plot(x, r, color=col, ls=ls, lw=1.5, label=lab)
    a.set_xlim(*xl); a.set_title(title); a.set_xlabel("x"); a.set_ylabel("densidad")
    m = (xf > xl[0]) & (xf < xl[1]); a.set_ylim(rf[m].min() - 0.3, rf[m].max() * 1.08)
    a.legend(fontsize=8, loc="lower left" if prob == "shuosher" else "upper left")

N = 50
e = np.linspace(0, 1, N + 1); ex = cell_avg_cons(PROBLEMS["entropy"]["ic"], e)[0]
xx = np.linspace(0, 1, 400); ax[3].plot(xx, 1 + 0.5 * np.sin(2 * np.pi * xx), "k", lw=0.8, label="exacta")
for n, col, ls, lab in show + [("H dc=0.3", "tab:cyan", "-", "híbrido mejorado (δ=0.3)")]:
    x, r, _, _ = solve("entropy", N, L[n], cfl=c)
    ax[3].plot(x, r, color=col, ls=ls, lw=1.5, marker=".", ms=4, label=f"{lab} (error {np.abs(r - ex).mean():.1e})")
ax[3].set_title("(d) Onda de entropía suave, N=50, una vuelta"); ax[3].set_xlabel("x"); ax[3].set_ylabel("densidad")
ax[3].legend(fontsize=8, loc="lower left"); ax[3].set_ylim(0.35, 1.6)
plt.tight_layout(); plt.savefig(os.path.join(_EXP, "img", "paso3.png"), dpi=120)
