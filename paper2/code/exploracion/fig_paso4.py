import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
_EXP = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(_EXP, '..'))  # datos .npy en ../
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from euler2d import run2d, prims, gresho_uphi_p, PROB2
from euler2d_tests import CONF

D = np.load("euler2d_res.npy", allow_pickle=True).item(); R = D["R"]
fn = lambda name: f"quad_{name.replace(' ', '_').replace('/', '')}_200.npy"
fig = plt.figure(figsize=(17, 10)); gs = GridSpec(2, 12, figure=fig, height_ratios=[1, 1.05])
show = [("Hibrido original (idea 2)", "tu híbrido original"), ("MC", "MC"),
        ("Kemm: CS lineales + AR t=.75", "Kemm: mejor combinación"), ("H dc=0.05", "híbrido mejorado (δ=0.05)")]
lev = np.linspace(0.15, 1.75, 30)
x = (np.arange(200) + 0.5) / 200
for k, (n, lab) in enumerate(show):
    a = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
    rho = np.load(fn(n))
    a.contour(x, x, rho.T, levels=lev, colors="k", linewidths=0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{lab}\nerror {R[(n, 'quad-refMC', 200)]:.2e}", fontsize=10)
fig.text(0.5, 0.965, "(a) Problema de Riemann 2D (Kemm §5.3), 200×200, t=0.8: contornos de densidad", ha="center", fontsize=12)

# (b) implosion: densidad vs radio
rr, dr = np.load("implosion_ref.npy")
a = fig.add_subplot(gs[1, 0:4])
for n, col, lab in [("MC", "0.6", "MC"), ("Kemm: CS lineales + AR t=.75", "tab:green", "Kemm: mejor combinación"), ("H dc=0.05", "tab:blue", "híbrido mejorado (δ=0.05)")]:
    X, Y, q = run2d("implosion", 100, CONF[n])
    a.plot(np.sqrt(X ** 2 + Y ** 2).ravel(), q[0].ravel(), ".", ms=1.5, color=col, label=f"{lab} (error {R[(n, 'impl', 100)]:.2e})")
a.plot(rr, dr, "k", lw=1, label="referencia radial")
a.set_xlim(0, 0.9); a.set_xlabel("radio"); a.set_ylabel("densidad")
a.set_title("(b) Implosión (Kemm §5.4), 100×100, t=0.25", fontsize=11); a.legend(fontsize=7.5, markerscale=5, loc="lower right")

# (c) Gresho: velocidad azimutal vs radio
a = fig.add_subplot(gs[1, 4:7])
rp = np.linspace(0, 0.6, 400); a.plot(rp, gresho_uphi_p(rp)[0], "k", lw=1, label="exacta")
for n, col, lab in [("MC", "0.6", "MC"), ("Kemm: CS lineales + AR t=.75", "tab:green", "Kemm: mejor combinación"), ("H dc=0.05", "tab:blue", "híbrido mejorado (δ=0.05)")]:
    X, Y, q = run2d("gresho", 40, CONF[n])
    rho, u, v, p = prims(q)
    dx = (X - 3.5 + 2) % 4 - 2; dy = Y - 0.5; r = np.sqrt(dx ** 2 + dy ** 2)
    uphi = (-(u - 1.0) * dy + v * dx) / np.maximum(r, 1e-12); m = r < 0.6
    a.plot(r[m], uphi[m], ".", ms=2, color=col, label=f"{lab} (error {R[(n, 'gresho', 40)]:.2e})")
a.set_xlabel("radio"); a.set_ylabel("velocidad azimutal")
a.set_title("(c) Vórtice de Gresho (Kemm §5.5), 160×40, t=3", fontsize=11); a.legend(fontsize=7.5, markerscale=4, loc="upper right"); a.set_ylim(-0.1, 1.45)

# (d) resumen: error relativo a MC
a = fig.add_subplot(gs[1, 9:12])
names = list(CONF); cols = [("quad-refMC", 200, "Riemann 2D"), ("impl", 200, "implosión 200²"), ("gresho", 40, "Gresho 160×40")]
lab = {"MC": "MC", "superbee": "Superbee", "MINBEE (idea 1)": "MINBEE (idea 1)", "Hibrido original (idea 2)": "híbrido original (idea 2)",
       "Kemm: AR t=.75 todas": "Kemm: AR θ=0.75", "Kemm: CS lineales + AR t=.75": "Kemm: mejor combinación",
       "H dc=0.05": "mejorado δ=0.05", "H dc=0.3": "mejorado δ=0.3", "H dc=1": "mejorado δ=1"}
y = np.arange(len(names)); w = 0.26
for j, (c, N, t) in enumerate(cols):
    vals = [R[(n, c, N)] / R[("MC", c, N)] for n in names]
    a.barh(y + (j - 1) * w, vals, height=w, label=t, color=["tab:gray", "tab:orange", "tab:purple"][j])
a.axvline(1, color="k", lw=0.8); a.set_yticks(y); a.set_yticklabels([lab[n] for n in names], fontsize=8); a.invert_yaxis()
a.set_xlabel("error relativo a MC (menor = mejor)"); a.set_title("(d) Resumen", fontsize=11); a.legend(fontsize=8)
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(os.path.join(_EXP, "img", "paso4.png"), dpi=110)
