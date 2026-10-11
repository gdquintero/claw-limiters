"""Figuras del borrador (ingles) -> paper/fig*.pdf"""
import numpy as np, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "legend.fontsize": 7.5, "axes.titlesize": 9.5})
from limlib import ub, minmod, superbee, mc, arora_roe, cfl_superbee
from ideas_check import minbee, topus, hybrid, omega
from hyb2 import omega_d, hyb_d
from hyb1 import go
from ideas_tests import cell_avg, zalesak

OUT = "../figures/"
H = lambda d: hyb_d(cfl_superbee, d)

# ---------- Fig 1: limiters and sensor ----------
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
r = np.linspace(0, 4, 801); nu = 0.5; nv = np.full_like(r, nu)
for a in ax[:2]:
    a.fill_between(r, 0, ub(r, nu), color="0.93", lw=0)
ax[0].fill_between(r, np.minimum(r, 1), np.minimum(np.minimum(2 * r, 2), np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2))), color="0.85", lw=0)
for f, lab, st in [(minbee, "MINBEE", "--"), (topus, "TOPUS", ":"), (hybrid, "original hybrid $H_0$", "-")]:
    ax[0].plot(r, f(r), st, lw=1.5, label=lab)
ax[0].set_title("(a) original ingredients (CFL-independent)")
for d, st in [(1, "-"), (0.3, "--"), (0.05, "-.")]:
    ax[1].plot(r, H(d)(r, nv), st, lw=1.5, label=fr"TOPBEE-$\delta$, $\delta$={d}")
ax[1].plot(r, 1 + (1 + nu) / 3 * (r - 1), "k", lw=0.7, label=r"$\Phi_3$ (third order)")
ax[1].set_title(r"(b) TOPBEE-$\delta$, $\nu=0.5$")
for a in ax[:2]:
    a.set_xlim(0, 4); a.set_ylim(0, 3); a.set_xlabel("$r$"); a.set_ylabel(r"$\Psi(r)$"); a.legend(loc="upper left")
rl = np.geomspace(1e-2, 1e2, 2001)
for d in [1, 0.3, 0.1, 0.03]:
    ax[2].semilogx(rl, omega_d(rl, d), lw=1.5, label=(r"$\delta=1$ ($\omega$, original sensor)" if d == 1 else fr"$\delta$={d}"))
ax[2].set_xlabel("$r$"); ax[2].set_ylabel(r"$\omega_\delta(r)$"); ax[2].set_title(r"(c) sensor $\omega_\delta$"); ax[2].legend()
plt.tight_layout(); plt.savefig(OUT + "fig1_limiters.pdf"); plt.close()

# ---------- Fig 2: von Neumann ----------
def G(xi, nu, b):
    E = np.exp(-1j * xi)
    return 1 - nu * (1 - E) - nu * (1 - nu) / 2 * (1 - E) * ((1 - b) * (np.exp(1j * xi) - 1) + b * (1 - E))
xi = np.linspace(1e-4, np.pi, 600)
fig, ax = plt.subplots(2, 2, figsize=(8.5, 5.4))
for j, nu in enumerate([0.2, 0.8]):
    for b, lab, st in [(0.25, r"$\beta=1/4$ (TOPUS, $H_0$)", ":"), (0.5, r"$\beta=1/2$ (Fromm, MINBEE)", "--"),
                       ((1 + nu) / 3, r"$\beta=(1+\nu)/3$ (Arora–Roe, TOPBEE-$\delta$)", "-"), (0.0, r"$\beta=0$ (Lax–Wendroff)", "-.")]:
        g = G(xi, nu, b)
        ax[0, j].plot(xi / np.pi, np.abs(g), st, lw=1.3, label=lab)
        ax[1, j].plot(xi / np.pi, -np.unwrap(np.angle(g)) / (nu * xi), st, lw=1.3, label=lab)
    ax[0, j].set_title(fr"amplification factor $|G|$, $\nu={nu}$"); ax[1, j].set_title(fr"relative phase speed, $\nu={nu}$")
    ax[1, j].set_xlabel(r"$\xi/\pi$"); ax[1, j].axhline(1, color="k", lw=0.5); ax[1, j].set_ylim(0, 1.6)
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + "fig2_vonneumann.pdf"); plt.close()

# ---------- Fig 3: linear advection ----------
R2 = np.load("hyb2_res.npy", allow_pickle=True).item()
D1 = np.load("hyb1_conv.npy", allow_pickle=True).item()
NUS = [0.1, 0.5, 0.9]
gm = lambda v: np.exp(np.mean(np.log(v)))
def pt2(n): return gm([R2[n][("zal", nu)] for nu in NUS]), gm([R2[n][("sin", nu, 800)] for nu in NUS])
def pt1(n): return gm([D1["res"][n][j][0] for j in range(3)]), gm([D1["conv"][n][nu][3] for nu in NUS])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
ds = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
P = [pt2(f"H3 d={d}") for d in ds]
ax[0].loglog(*zip(*P), "-o", ms=3.5, color="tab:blue", label=r"TOPBEE-$\delta$")
for d, p in zip(ds, P): ax[0].annotate(str(d), p, textcoords="offset points", xytext=(3, -9), fontsize=6.5, color="tab:blue")
for n, lab, m, c in [("H0 original", "$H_0$", "s", "tab:red"), ("Arora-Roe", "Arora–Roe", "^", "0.3"),
                     ("Kemm beta=2/3", r"Kemm $\beta=2/3$", "D", "tab:green"), ("CFL-Superbee", "CFL-Superbee", "v", "tab:purple")]:
    ax[0].loglog(*pt2(n), m, color=c, label=lab)
for n, lab, m, c in [("MINBEE", "MINBEE", "P", "tab:orange"), ("MC", "MC", "X", "k")]:
    ax[0].loglog(*pt1(n), m, color=c, label=lab)
ax[0].set_xlabel("Zalesak $L_1$ error (N=200, t=20)"); ax[0].set_ylabel(r"$\sin 6\pi x$ $L_1$ error (N=800)")
ax[0].set_title("(a) accuracy trade-off (geo. mean over $\\nu$=0.1,0.5,0.9)"); ax[0].legend(fontsize=6.5, loc="lower left")
N = 200; e = np.linspace(-1, 1, N + 1); x = 0.5 * (e[1:] + e[:-1])
qw = cell_avg(lambda x: np.exp(-50 * x ** 2) * np.sin(20 * np.pi * x), e)
ax[1].plot(x, qw, "k", lw=0.8, label="exact")
from hyb2 import LIM as L2
for n, lab, st in [("H0 original", "$H_0$", "-"), ("Kemm beta=2/3", r"Kemm $\beta=2/3$", "--"), ("H3 d=0.05", r"TOPBEE-$\delta$, $\delta=0.05$", "-")]:
    ax[1].plot(x, go(qw, 0.5, 10, 2 / N, L2[n]), st, lw=1.2, label=lab)
ax[1].set_xlim(-0.5, 0.5); ax[1].set_title(r"(b) wave packet, N=200, t=10, $\nu=0.5$"); ax[1].legend(fontsize=6.5, loc="lower left")
Ns = [50, 100, 200, 400, 800, 1600, 3200, 6400]
for n, lab, st in [("H0 original", "$H_0$", "s-"), ("H3 d=1", r"TOPBEE-$\delta$, $\delta=1$", "o-"), ("H3 d=0.05", r"TOPBEE-$\delta$, $\delta=0.05$", "o--"),
                   ("Arora-Roe", "Arora–Roe", "^:"), ("Kemm beta=2/3", r"Kemm $\beta=2/3$", "D-.")]:
    ax[2].loglog(Ns, [R2[n][("sin", 0.9, k)] for k in Ns], st, ms=3, lw=1, label=lab)
ax[2].loglog(D1["Ns"], D1["conv"]["MINBEE"][0.9], "P-", ms=3, lw=1, label="MINBEE")
ax[2].loglog(Ns, 2e-1 * (np.array(Ns) / 50.) ** -2, color="0.7", lw=0.6); ax[2].loglog(Ns, 2e-2 * (np.array(Ns) / 50.) ** -3, color="0.7", lw=0.6)
ax[2].set_xlabel("N"); ax[2].set_ylabel("$L_1$ error"); ax[2].set_title(r"(c) convergence, $\sin 6\pi x$, $\nu=0.9$"); ax[2].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(OUT + "fig3_advection.pdf"); plt.close()

# ---------- Fig 4: Euler 1D ----------
from euler1d import solve
from euler_pareto import L as LE, DC
RE = np.load("euler_pareto.npy", allow_pickle=True).item()["R"]
CASES = [("sod", 200), ("lax", 200), ("shuosher", 400), ("blast", 400)]
c = 0.9
sh = lambda n: np.exp(np.mean([np.log(RE[(n, p, c)] / RE[("MC", p, c)]) for p, N in CASES]))
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
hs = [f"H dc={d}" for d in DC]
ax[0].loglog([sh(n) for n in hs], [RE[(n, "ent", c, 200)] for n in hs], "-o", ms=3.5, color="tab:blue", label=r"TOPBEE-$\delta$ (varying $\delta_c$)")
for n in hs: ax[0].annotate(n.split("=")[1], (sh(n), RE[(n, "ent", c, 200)]), textcoords="offset points", xytext=(4, -3), fontsize=6.5, color="tab:blue")
for n, lab, m, col in [("MC", "MC", "X", "k"), ("superbee", "Superbee", "v", "tab:purple"), ("MINBEE (idea 1)", "MINBEE", "P", "tab:orange"),
                       ("Hibrido original (idea 2)", "$H_0$", "s", "tab:red"), ("Kemm: AR t=.75 todas", r"Kemm: AR$_{0.75}$ all waves", "^", "tab:olive"),
                       ("Kemm: CS contacto + AR t=.75", r"Kemm: CS contact + AR$_{0.75}$", "D", "tab:green")]:
    ax[0].loglog(sh(n), RE[(n, "ent", c, 200)], m, color=col, label=lab)
ax[0].set_xlabel("shock-test error relative to MC"); ax[0].set_ylabel("entropy-wave $L_1$ error (N=200)")
ax[0].set_title(r"(a) Euler 1D trade-off, CFL 0.9"); ax[0].legend(fontsize=6.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
show = [("Hibrido original (idea 2)", "$H_0$", "-"), ("MC", "MC", ":"), ("Kemm: CS contacto + AR t=.75", r"Kemm: CS contact + AR$_{0.75}$", "--"), ("H dc=0.05", r"TOPBEE-$\delta$, $\delta_c=0.05$", "-")]
for k, (prob, N, xl) in enumerate([("shuosher", 400, (0.4, 2.6)), ("blast", 400, (0.6, 0.86))]):
    a = ax[1 + k]
    xf, rf, _, _ = solve(prob, 12800, fortran=True, fortran_lim=4, cfl=0.8)
    a.plot(xf, rf, "k", lw=0.7, label="reference")
    for n, lab, st in show:
        xx, rr, _, _ = solve(prob, N, LE[n], cfl=c); a.plot(xx, rr, st, lw=1.1, label=lab)
    m = (xf > xl[0]) & (xf < xl[1]); a.set_xlim(*xl); a.set_ylim(rf[m].min() - 0.3, rf[m].max() * 1.08)
    a.set_title(["(b) Shu–Osher, N=400, t=1.8", "(c) Woodward–Colella, N=400, t=0.038"][k]); a.set_xlabel("x"); a.set_ylabel("density")
    a.legend(fontsize=6.3, loc="lower left" if k == 0 else "upper left")
plt.tight_layout(); plt.subplots_adjust(bottom=0.33); plt.savefig(OUT + "fig4_euler1d.pdf"); plt.close()

# ---------- Fig 5: Euler 2D ----------
from euler2d import run2d, prims, gresho_uphi_p
from euler2d_tests import CONF
R2D = np.load("euler2d_res.npy", allow_pickle=True).item()["R"]
fn = lambda name: f"quad_{name.replace(' ', '_').replace('/', '')}_200.npy"
fig = plt.figure(figsize=(10.5, 6.2)); gsp = fig.add_gridspec(2, 4, height_ratios=[1, 0.95])
lev = np.linspace(0.15, 1.75, 30); xq = (np.arange(200) + 0.5) / 200
for k, (n, lab) in enumerate([("Hibrido original (idea 2)", "$H_0$"), ("MC", "MC"), ("Kemm: CS lineales + AR t=.75", r"Kemm: CS lin. + AR$_{0.75}$"), ("H dc=0.05", r"TOPBEE-$\delta$, $\delta_c$=0.05")]):
    a = fig.add_subplot(gsp[0, k]); a.contour(xq, xq, np.load(fn(n)).T, levels=lev, colors="k", linewidths=0.35)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(f"({'abcd'[k]}) {lab}")
rr, dr = np.load("implosion_ref.npy")
a = fig.add_subplot(gsp[1, 0:2])
for n, lab, col in [("MC", "MC", "0.6"), ("Kemm: CS lineales + AR t=.75", r"Kemm: CS lin. + AR$_{0.75}$", "tab:green"), ("H dc=0.05", r"TOPBEE-$\delta$, $\delta_c$=0.05", "tab:blue")]:
    X, Y, q = run2d("implosion", 100, CONF[n]); a.plot(np.sqrt(X ** 2 + Y ** 2).ravel(), q[0].ravel(), ".", ms=1, color=col, label=lab)
a.plot(rr, dr, "k", lw=0.8, label="1D radial reference"); a.set_xlim(0, 0.9); a.set_xlabel("radius"); a.set_ylabel("density")
a.set_title("(e) implosion, 100×100, t=0.25"); a.legend(markerscale=6, loc="lower right")
a = fig.add_subplot(gsp[1, 2:4]); rp = np.linspace(0, 0.6, 300); a.plot(rp, gresho_uphi_p(rp)[0], "k", lw=0.8, label="exact")
for n, lab, col in [("MC", "MC", "0.6"), ("Kemm: CS lineales + AR t=.75", r"Kemm: CS lin. + AR$_{0.75}$", "tab:green"), ("H dc=0.05", r"TOPBEE-$\delta$, $\delta_c$=0.05", "tab:blue")]:
    X, Y, q = run2d("gresho", 40, CONF[n]); rho, u, v, p = prims(q)
    dx = (X - 3.5 + 2) % 4 - 2; dy = Y - 0.5; rad = np.sqrt(dx ** 2 + dy ** 2); m = rad < 0.6
    a.plot(rad[m], ((-(u - 1) * dy + v * dx) / np.maximum(rad, 1e-12))[m], ".", ms=1.2, color=col, label=lab)
a.set_xlabel("radius"); a.set_ylabel(r"$u_\varphi$"); a.set_title("(f) Gresho vortex, 160×40, t=3"); a.legend(markerscale=6); a.set_ylim(-0.1, 1.35)
plt.tight_layout(); plt.savefig(OUT + "fig5_euler2d.pdf"); plt.close()
print("figuras listas")
