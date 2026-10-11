import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
_EXP = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(_EXP, '..'))  # datos .npy en ../
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyb1 import LIM, go
from limlib import ub
from ideas_tests import cell_avg, zalesak

D = np.load("hyb1_conv.npy", allow_pickle=True).item()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
sty = {"H0 original": ("tab:red", "-", "Tu híbrido original (MINBEE + TOPUS)"),
       "H1 MINBEE+AR": ("tab:orange", "--", "Cambio 1: TOPUS → parte de 3er orden"),
       "H3 CFLSB+AR": ("tab:blue", "-", "Cambio 1 + 2: además parte compresiva según CFL")}

# (a) curvas del limitador, nu = 0.5
nu = 0.5; r = np.linspace(0, 4, 801); nv = np.full_like(r, nu)
ax[0].fill_between(r, 0, ub(r, nu), color="0.92", label="zona permitida (TVD)")
ax[0].plot(r, np.minimum(1, r), color="0.5", lw=0.8, ls=":")
for k, (c, ls, lab) in sty.items():
    ax[0].plot(r, LIM[k](r, nv), color=c, ls=ls, lw=2, label=lab)
ax[0].plot(r, 1 + (1 + nu) / 3 * (r - 1), color="k", lw=0.8, ls="-.", label="recta de 3er orden")
ax[0].set_xlim(0, 4); ax[0].set_ylim(0, 3); ax[0].set_xlabel("r"); ax[0].set_ylabel("Ψ(r)")
ax[0].set_title("(a) Forma del limitador (CFL 0.5)"); ax[0].legend(fontsize=8, loc="upper left")

# (b) Zalesak t=20, nu=0.5
N = 200; e = np.linspace(-1, 1, N + 1); x = 0.5 * (e[1:] + e[:-1]); q0 = cell_avg(zalesak, e)
ax[1].plot(x, q0, color="k", lw=1, label="exacta")
for k, (c, ls, lab) in sty.items():
    ax[1].plot(x, go(q0, 0.5, 20, 2 / N, LIM[k]), color=c, ls=ls, lw=1.6,
               label=f"{lab.split(':')[0]} (error {D['res'][k][1][0]:.3f})")
ax[1].set_title("(b) Prueba de Zalesak, 10 vueltas (N=200, CFL 0.5)"); ax[1].set_xlabel("x")
ax[1].legend(fontsize=8, loc="upper right"); ax[1].set_ylim(-0.1, 1.25)

# (c) convergencia, nu = 0.9
Ns = np.array(D["Ns"])
for k, (c, ls, lab) in sty.items():
    ax[2].loglog(Ns, D["conv"][k][0.9], color=c, ls=ls, marker="o", lw=1.6, label=lab.split(":")[0])
ax[2].loglog(Ns, D["conv"]["Arora-Roe"][0.9], color="0.4", ls=":", marker="s", ms=4, label="Arora–Roe (referencia)")
ax[2].loglog(Ns, 3e-1 * (Ns / 100.) ** -2, color="0.7", lw=0.8); ax[2].text(1600, 3e-1 * 16 ** -2 * 1.3, "pendiente 2", color="0.5", fontsize=8)
ax[2].loglog(Ns, 3e-3 * (Ns / 100.) ** -3, color="0.7", lw=0.8); ax[2].text(1600, 3e-3 * 16 ** -3 * 0.4, "pendiente 3", color="0.5", fontsize=8)
ax[2].set_xlabel("N (número de celdas)"); ax[2].set_ylabel("error L1")
ax[2].set_title("(c) Solución suave sin(6πx): error vs malla (CFL 0.9)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(_EXP, "img", "paso1.png"), dpi=130)
