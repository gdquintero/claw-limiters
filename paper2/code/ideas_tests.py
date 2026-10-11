"""Linear advection q_t + q_x = 0 on [-1,1], periodic. Two frameworks:
 (A) one-step flux-limiter / wave-propagation (LeVeque eq. 6.40, = CLAWPACK 1D for advection)
 (B) semi-discrete MUSCL face value q_f = q_U + 0.5*Psi(r)*(q_D - q_U) + SSP-RK3
"""
import numpy as np
from ideas_check import superbee, minmod, mc, vanleer, minbee, topus, hybrid, koren3

def arora_roe(nu):
    return lambda r: np.maximum(0, np.minimum(np.minimum(2*r/nu, 1 + (1 + nu)/3*(r - 1)), 2/(1 - nu)))

def ratio(dq_up, dq_loc):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(dq_loc != 0, dq_up/np.where(dq_loc != 0, dq_loc, 1), 0.0)
    return r

def step_onestep(q, nu, psi):
    dq = np.roll(q, -1) - q                    # dq[i] = q[i+1]-q[i] = jump at i+1/2
    r = ratio(np.roll(dq, 1), dq)              # upwind jump / local jump at i+1/2
    F = q + 0.5*(1 - nu)*psi(r)*dq             # flux at i+1/2 (a=1), divided by a
    return q - nu*(F - np.roll(F, 1))

def L_semi(q, psi):
    dq = np.roll(q, -1) - q
    r = ratio(np.roll(dq, 1), dq)
    F = q + 0.5*psi(r)*dq
    return -(F - np.roll(F, 1))                # times 1/dx applied outside

def step_ssprk3(q, nu, psi):
    q1 = q + nu*L_semi(q, psi)
    q2 = 0.75*q + 0.25*(q1 + nu*L_semi(q1, psi))
    return q/3 + 2/3*(q2 + nu*L_semi(q2, psi))

def cell_avg(f, x_edges, m=16):
    g, w = np.polynomial.legendre.leggauss(m)
    a, b = x_edges[:-1], x_edges[1:]
    xm, xr = 0.5*(a + b), 0.5*(b - a)
    return sum(wi*f(xm + xr*gi) for gi, wi in zip(g, w))/2

def zalesak(x):
    a, z, d, al = 0.5, -0.7, 0.005, 10.0
    be = np.log(2)/(36*d*d)
    G = lambda x, z: np.exp(-be*(x - z)**2)
    F = lambda x, a: np.sqrt(np.maximum(1 - al**2*(x - a)**2, 0))
    u = np.zeros_like(x)
    m = (x >= -0.8) & (x <= -0.6); u[m] = (G(x[m], z - d) + G(x[m], z + d) + 4*G(x[m], z))/6
    m = (x >= -0.4) & (x <= -0.2); u[m] = 1.0
    m = (x >= 0.0) & (x <= 0.2); u[m] = 1 - np.abs(10*(x[m] - 0.1))
    m = (x >= 0.4) & (x <= 0.6); u[m] = (F(x[m], a - d) + F(x[m], a + d) + 4*F(x[m], a))/6
    return u

def run(q0, N, nu, T, psi, framework):
    dx = 2.0/N; dt = nu*dx; n = int(round(T/dt)); q = q0.copy()
    stepper = step_onestep if framework == 'A' else step_ssprk3
    for _ in range(n): q = stepper(q, nu, psi)
    return q

if __name__ == "__main__":
    lims = {"minmod": minmod, "vanLeer": vanleer, "MC": mc, "superbee": superbee,
            "MINBEE": minbee, "TOPUS": topus, "HYBRID": hybrid, "Koren(k=1/3)": koren3}
    N = 200; e = np.linspace(-1, 1, N + 1); q0 = cell_avg(zalesak, e)
    for fw, nu in [("A", 0.5), ("A", 0.9), ("B", 0.4)]:
        L = dict(lims)
        if fw == "A": L["Arora-Roe(CFL)"] = arora_roe(nu)
        print(f"\n=== Zalesak, N=200, framework {fw} ({'one-step/CLAWPACK' if fw=='A' else 'semi-discrete + SSP-RK3'}), nu={nu}")
        print(f"{'limiter':16s} {'L1 t=2':>9s} {'L1 t=20':>9s} {'max t=20':>9s} {'min t=20':>9s}")
        for name, psi in L.items():
            q2 = run(q0, N, nu, 2.0, psi, fw); q20 = run(q0, N, nu, 20.0, psi, fw)
            print(f"{name:16s} {np.abs(q2-q0).sum()*2/N:9.4f} {np.abs(q20-q0).sum()*2/N:9.4f} {q20.max():9.4f} {q20.min():9.2e}")
    # convergence: sin(6 pi x), T=2 (one period)
    f = lambda x: np.sin(6*np.pi*x)
    for fw, nu in [("A", 0.5), ("B", 0.4)]:
        L = dict(lims)
        if fw == "A": L["Arora-Roe(CFL)"] = arora_roe(nu)
        print(f"\n=== sin(6 pi x), T=2, L1 error / observed order, framework {fw}, nu={nu}")
        Ns = [100, 200, 400, 800, 1600]
        print(f"{'limiter':16s}" + "".join(f"{n:>11d}" for n in Ns))
        for name, psi in L.items():
            errs = []
            for n in Ns:
                e = np.linspace(-1, 1, n + 1); q0n = cell_avg(f, e)
                errs.append(np.abs(run(q0n, n, nu, 2.0, psi, fw) - q0n).sum()*2/n)
            s = f"{name:16s}" + f"{errs[0]:11.2e}"
            for k in range(1, len(Ns)):
                s += f" {errs[k]:.1e}({np.log2(errs[k-1]/errs[k]):.2f})"
            print(s)
