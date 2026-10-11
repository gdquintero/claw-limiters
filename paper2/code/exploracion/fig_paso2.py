import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
_EXP = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(_EXP, '..'))  # datos .npy en ../
import numpy as np, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyb2 import LIM, DELTAS, omega_d, NUS
from hyb1 import go
from ideas_tests import cell_avg, zalesak

R = np.load("hyb2_res.npy", allow_pickle=True).item()
DSEL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
NSM = int(sys.argv[2]) if len(sys.argv) > 2 else 400
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5)); ax = ax.ravel()

# (a) sensor
r = np.geomspace(1e-2, 1e2, 2001)
for d, c in zip([1, 0.3, 0.1, 0.03], ["tab:red", "tab:orange", "tab:blue", "tab:purple"]):
    ax[0].semilogx(r, omega_d(r, d), color=c, lw=2, label=("δ = 1 (tu sensor original)" if d == 1 else f"δ = {d}"))
ax[0].set_xlabel("r  (escala logarítmica)"); ax[0].set_ylabel("peso de la parte suave ω_δ")
ax[0].set_title("(a) Tu sensor con umbral δ"); ax[0].legend(fontsize=9)

# (b) mapa error-discontinuidad vs error-suave (promedio geometrico sobre los 3 CFL)
gm = lambda v: np.exp(np.mean(np.log(v)))
X = {n: gm([R[n][("zal", nu)] for nu in NUS]) for n in R}
Y = {n: gm([R[n][("sin", nu, NSM)] for nu in NUS]) for n in R}
h3 = [f"H3 d={d}" for d in DELTAS]
ax[1].loglog([X[n] for n in h3], [Y[n] for n in h3], "-o", color="tab:blue", label="híbrido mejorado, variando δ")
for n in h3:
    ax[1].annotate(n.split("=")[1], (X[n], Y[n]), textcoords="offset points", xytext=(4, -11), fontsize=8, color="tab:blue")
for n, c, m, lab in [("H0 original", "tab:red", "s", "tu híbrido original"), ("Arora-Roe", "0.3", "^", "Arora–Roe"),
                     ("Kemm beta=2/3", "tab:green", "D", "Kemm β=2/3 (mejor de Kemm)"), ("CFL-Superbee", "tab:purple", "v", "CFL-Superbee")]:
    ax[1].loglog(X[n], Y[n], m, color=c, ms=9, label=lab)
ax[1].set_xlabel("error en discontinuidades (Zalesak)"); ax[1].set_ylabel(f"error en solución suave (sin 6πx, N={NSM})")
ax[1].set_title("(b) Abajo a la izquierda = mejor\n(promedio de CFL 0.1, 0.5, 0.9)"); ax[1].legend(fontsize=8, loc="lower left"); ax[1].set_xlim(0.03, 0.4)

# (c) Zalesak
N = 200; e = np.linspace(-1, 1, N + 1); x = 0.5 * (e[1:] + e[:-1]); q0 = cell_avg(zalesak, e)
nu = 0.5
ax[2].plot(x, q0, "k", lw=1, label="exacta")
for n, c, ls, lab in [("H0 original", "tab:red", "-", "tu híbrido original"), ("Kemm beta=2/3", "tab:green", "--", "Kemm β=2/3"),
                      (f"H3 d={DSEL:g}", "tab:blue", "-", f"híbrido mejorado δ={DSEL:g}")]:
    ax[2].plot(x, go(q0, nu, 20, 2 / N, LIM[n]), color=c, ls=ls, lw=1.6, label=f"{lab} (error {R[n][('zal', nu)]:.3f})")
ax[2].set_ylim(-0.1, 1.25); ax[2].set_xlabel("x"); ax[2].legend(fontsize=8, loc="upper right")
ax[2].set_title("(c) Zalesak, 10 vueltas (N=200, CFL 0.5)")

qw = cell_avg(lambda x: np.exp(-50 * x ** 2) * np.sin(20 * np.pi * x), e)
ax[3].plot(x, qw, "k", lw=1, label="exacta")
for n, c, ls, lab in [("H0 original", "tab:red", "-", "tu híbrido original"), ("Kemm beta=2/3", "tab:green", "--", "Kemm β=2/3"),
                      (f"H3 d={DSEL:g}", "tab:blue", "-", f"híbrido mejorado δ={DSEL:g}")]:
    ax[3].plot(x, go(qw, nu, 10, 2 / N, LIM[n]), color=c, ls=ls, lw=1.6, label=f"{lab} (error {R[n][('wp', nu)]:.3f})")
ax[3].set_xlim(-0.6, 0.6); ax[3].set_ylim(-1.15, 1.45); ax[3].set_xlabel("x"); ax[3].legend(fontsize=8, loc="upper right")
ax[3].set_title("(d) Paquete de ondas, 5 vueltas (N=200, CFL 0.5)")
plt.tight_layout(); plt.savefig(os.path.join(_EXP, "img", "paso2.png"), dpi=120)
