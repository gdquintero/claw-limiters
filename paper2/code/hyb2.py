"""Paso 2 (paper 2): sensor del usuario con umbral delta.
  w_delta = delta^2 w / (1 - w + delta^2 w),   w = (2|r|/(1+r^2))^2   (delta=1 -> sensor original)
  psi = (1 - w_delta) C + w_delta S,  S = Arora-Roe,  C = CFL-Superbee (H3) o CFL-MINBEE (H2)
"""
import numpy as np, sys
from limlib import arora_roe, cfl_superbee, beta_lim
from ideas_check import omega
from ideas_tests import cell_avg, zalesak
from hyb1 import go, cfl_minbee, LIM as LIM1
from limprops import props


def omega_d(r, delta):
    w = omega(r)
    return delta ** 2 * w / (1 - w + delta ** 2 * w)


def hyb_d(C, delta):
    def f(r, nu):
        w = omega_d(r, delta)
        return (1 - w) * C(r, nu) + w * arora_roe(r, nu)
    return f


DELTAS = [1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
LIM = {}
for d in DELTAS:
    LIM[f"H3 d={d}"] = hyb_d(cfl_superbee, d)
for d in [1, 0.2, 0.1, 0.05]:
    LIM[f"H2 d={d}"] = hyb_d(cfl_minbee, d)
for k in ["H0 original", "Arora-Roe", "Kemm beta=2/3"]:
    LIM[k] = LIM1[k]
LIM["CFL-Superbee"] = cfl_superbee

NUS = [0.1, 0.5, 0.9]


def tests(f):
    out = {}
    N = 200; e = np.linspace(-1, 1, N + 1); dx = 2 / N
    qz = cell_avg(zalesak, e)
    qw = cell_avg(lambda x: np.exp(-50 * x ** 2) * np.sin(20 * np.pi * x), e)
    for nu in NUS:
        out[("zal", nu)] = np.abs(go(qz, nu, 20, dx, f) - qz).sum() * dx
        out[("wp", nu)] = np.abs(go(qw, nu, 10, dx, f) - qw).sum() * dx
        for Ns in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
            ee = np.linspace(-1, 1, Ns + 1); q0 = cell_avg(lambda x: np.sin(6 * np.pi * x), ee)
            out[("sin", nu, Ns)] = np.abs(go(q0, nu, 2, 2 / Ns, f) - q0).sum() * 2 / Ns
    return out


if __name__ == "__main__":
    print("=== propiedades (TVD, M2, tangencia de 3er orden)")
    for n, f in LIM.items():
        s = []
        for nu in NUS:
            t, rs, i, m, v1, sl, sr, tg = props(f, nu)
            s.append(f"nu={nu}: TVD={t!s:5s} M2={i!s:5s} M5={m!s:5s} {tg}")
        print(f"  {n:14s} " + " | ".join(s))
    R = {}
    for n, f in LIM.items():
        R[n] = tests(f); print(n, "ok", file=sys.stderr)
    np.save("hyb2_res.npy", R, allow_pickle=True)
    for lab, key in [("Zalesak L1 (N=200,t=20)", lambda nu: ("zal", nu)),
                     ("paquete de ondas L1 (N=200,t=10)", lambda nu: ("wp", nu)),
                     ("sin6pi L1 N=100", lambda nu: ("sin", nu, 100)),
                     ("sin6pi L1 N=800", lambda nu: ("sin", nu, 800)),
                     ("sin6pi L1 N=6400", lambda nu: ("sin", nu, 6400))]:
        print(f"\n=== {lab}")
        for n in LIM:
            print(f"  {n:14s}" + "".join(f"  nu={nu}: {R[n][key(nu)]:.2e}" for nu in NUS))
    print("\n=== orden observado sin6pi (1600->6400)")
    for n in LIM:
        print(f"  {n:14s}" + "".join(f"  nu={nu}: {np.log2(R[n][('sin', nu, 1600)] / R[n][('sin', nu, 6400)]) / 2:.2f}" for nu in NUS))
