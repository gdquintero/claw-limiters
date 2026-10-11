"""Paso 1 (paper 2): hibrido del usuario  psi = (1-w) C + w S,  w(r) = (2|r|/(1+r^2))^2  (sin cambios),
con ingredientes potenciados para el marco de un paso (CLAWPACK):
  H0 = original      C = MINBEE,      S = TOPUS
  H1                 C = MINBEE,      S = Arora-Roe (tercer orden, depende del CFL)
  H2                 C = CFL-MINBEE,  S = Arora-Roe     (CFL-MINBEE = 1/2 (CFL-Superbee + Minmod))
  H3                 C = CFL-Superbee, S = Arora-Roe
"""
import numpy as np, sys
from limlib import arora_roe, cfl_superbee, minmod, mc, beta_lim, ub, phi3
from ideas_check import minbee, topus, omega, hybrid
from ideas_tests import cell_avg, zalesak
from advtests import run
from limprops import props


def go(q0, nu, T, dx, f):
    """avanza hasta T exacto: numero entero de pasos, nu ajustado (<= nu pedido)"""
    n = int(np.ceil(T / (nu * dx) - 1e-9))
    return run(q0, T / (n * dx), n, f)


def cfl_minbee(r, nu):
    return 0.5 * (cfl_superbee(r, nu) + minmod(r))


def hyb(C, S):
    return lambda r, nu: (1 - omega(r)) * C(r, nu) + omega(r) * S(r, nu)


cfl_free = lambda f: (lambda r, nu: f(r))
LIM = {
    "MINBEE": cfl_free(minbee),
    "TOPUS": cfl_free(topus),
    "H0 original": cfl_free(hybrid),
    "H1 MINBEE+AR": hyb(cfl_free(minbee), arora_roe),
    "H2 CFLMINBEE+AR": hyb(cfl_minbee, arora_roe),
    "H3 CFLSB+AR": hyb(cfl_superbee, arora_roe),
    "CFL-MINBEE": cfl_minbee,
    "MC": mc,
    "Arora-Roe": arora_roe,
    "Kemm beta=2/3": beta_lim,
}

if __name__ == "__main__":
    nus = [0.1, 0.5, 0.9]
    print("=== propiedades")
    for nu in nus:
        print(f"nu={nu}: pendiente 3er orden (1+nu)/3={(1 + nu) / 3:.3f}")
        for n, f in LIM.items():
            t, rs, i, m, v1, sl, sr, tg = props(f, nu)
            print(f"  {n:16s} TVD={t!s:5s} M2={i!s:5s} psi(1)={v1:.3f} pend={sl:.3f}/{sr:.3f} tangencia={tg}")

    N = 200; e = np.linspace(-1, 1, N + 1); dx = 2 / N
    qz = cell_avg(zalesak, e)
    qw = cell_avg(lambda x: np.exp(-50 * x ** 2) * np.sin(20 * np.pi * x), e)
    res = {}
    for n, f in LIM.items():
        row = []
        for nu in nus:
            ez = np.abs(go(qz, nu, 20, dx, f) - qz).sum() * dx
            ew = np.abs(go(qw, nu, 10, dx, f) - qw).sum() * dx
            row.append((ez, ew))
        res[n] = row
        print(n, "ok", file=sys.stderr)
    for k, lab in enumerate(["Zalesak (N=200, t=20)", "paquete de ondas (N=200, t=10)"]):
        print(f"\n=== error L1, {lab}")
        for n in sorted(res, key=lambda n: sum(res[n][j][k] for j in range(3))):
            print(f"  {n:16s}" + "".join(f"  nu={nu}: {res[n][j][k]:.2e}" for j, nu in enumerate(nus)))

    print("\n=== convergencia sin(6 pi x), T=2: error L1 en N=3200 y orden observado (800->3200)")
    Ns = [100, 200, 400, 800, 1600, 3200]
    conv = {}
    for n, f in LIM.items():
        conv[n] = {}
        for nu in nus:
            errs = []
            for Ns_ in Ns:
                ee = np.linspace(-1, 1, Ns_ + 1); q0 = cell_avg(lambda x: np.sin(6 * np.pi * x), ee)
                errs.append(np.abs(go(q0, nu, 2, 2 / Ns_, f) - q0).sum() * 2 / Ns_)
            conv[n][nu] = errs
        print(f"  {n:16s}" + "".join(f"  nu={nu}: {conv[n][nu][-1]:.1e} (orden {np.log2(conv[n][nu][-3] / conv[n][nu][-1]) / 2:.2f})" for nu in nus))
    np.save("hyb1_conv.npy", {"Ns": Ns, "conv": conv, "res": res}, allow_pickle=True)
