"""Paso 3: Euler 1D en PyClaw (CLAWPACK, solver clasico de un paso, kernel Python) con limitadores propios.
Riemann: Roe (riemann.euler_1D_py) + correccion de entropia de Harten-Hyman (igual que rp1_euler_with_efix.f90).
"""
import numpy as np
from clawpack import pyclaw, riemann
from clawpack.pyclaw.limiters import tvd
from clawpack.riemann.euler_1D_py import euler_roe_1D

GAMMA = 1.4
PD = {"gamma": GAMMA, "gamma1": GAMMA - 1, "efix": False}


# ---------------- Riemann solver: Roe + Harten-Hyman entropy fix ----------------
def _p_c(rho, mom, E):
    p = (GAMMA - 1) * (E - 0.5 * mom ** 2 / rho)
    return p, np.sqrt(GAMMA * np.maximum(p, 1e-300) / rho)


def roe_efix(q_l, q_r, aux_l, aux_r, pd):
    wave, s, _, _ = euler_roe_1D(q_l, q_r, aux_l, aux_r, pd)
    pl, cl = _p_c(*q_l); ul = q_l[1] / q_l[0]
    pr, cr = _p_c(*q_r); ur = q_r[1] / q_r[0]
    df = np.einsum("mwi,wi->mi", wave, s)                     # = f(q_r) - f(q_l)
    # 1-wave
    s0 = ul - cl
    q1 = q_l + wave[:, 0, :]
    p1, c1 = _p_c(*q1); s1 = q1[1] / q1[0] - c1
    sf1 = np.where((s0 < 0) & (s1 > 0), s0 * (s1 - s[0]) / np.where(s1 - s0 != 0, s1 - s0, 1), np.minimum(s[0], 0))
    # 3-wave
    q2 = q_r - wave[:, 2, :]
    p2, c2 = _p_c(*q2); s2 = q2[1] / q2[0] + c2
    s3 = ur + cr
    sf3 = np.where((s2 < 0) & (s3 > 0), s2 * (s3 - s[2]) / np.where(s3 - s2 != 0, s3 - s2, 1), np.minimum(s[2], 0))
    amdq = sf1 * wave[:, 0, :] + np.minimum(s[1], 0) * wave[:, 1, :] + sf3 * wave[:, 2, :]
    # caso totalmente supersonico hacia la derecha (Fortran: si s0>=0 y s(1)>0 => amdq=0)
    amdq = np.where((s0 >= 0) & (s[0] > 0), 0.0, amdq)
    return wave, s, amdq, df - amdq


# ---------------- limitadores ----------------
_REG = {}


def register(name, f):
    """registra f(r, cfl) en la tabla de PyClaw y devuelve su numero"""
    if name not in _REG:
        k = 200 + len(_REG)
        tvd.limiter_functions[k] = f
        _REG[name] = k
    return _REG[name]


# ---------------- problemas ----------------
def prim2cons(rho, u, p):
    return np.array([rho, rho * u, p / (GAMMA - 1) + 0.5 * rho * u ** 2])


def cell_avg_cons(prim, edges, m=8):
    g, w = np.polynomial.legendre.leggauss(m)
    a, b = edges[:-1], edges[1:]
    xm, xr = 0.5 * (a + b), 0.5 * (b - a)
    return sum(wi * prim2cons(*prim(xm + xr * gi)) for gi, wi in zip(g, w)) / 2


def riemann_ic(L, R, x0):
    return lambda x: tuple(np.where(x < x0, a, b) for a, b in zip(L, R))


PROBLEMS = {
    "sod":   dict(a=0.0, b=1.0, t=0.2, bc="extrap", ic=riemann_ic((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.5)),
    "lax":   dict(a=0.0, b=1.0, t=0.14, bc="extrap", ic=riemann_ic((0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.5)),
    "shuosher": dict(a=-5.0, b=5.0, t=1.8, bc="extrap",
                     ic=lambda x: (np.where(x < -4, 3.857143, 1 + 0.2 * np.sin(5 * x)),
                                   np.where(x < -4, 2.629369, 0.0), np.where(x < -4, 10.33333, 1.0))),
    "entropy": dict(a=0.0, b=1.0, t=1.0, bc="periodic",
                    ic=lambda x: (1 + 0.5 * np.sin(2 * np.pi * x), np.ones_like(x), np.ones_like(x))),
    "blast": dict(a=0.0, b=1.0, t=0.038, bc="wall",
                  ic=lambda x: (np.ones_like(x), np.zeros_like(x), np.where(x < 0.1, 1000.0, np.where(x < 0.9, 0.01, 100.0)))),
}


def solve(prob, N, limiter=None, cfl=0.9, fortran=False, fortran_lim=4):
    P = PROBLEMS[prob]
    if fortran:
        solver = pyclaw.ClawSolver1D(riemann.euler_with_efix_1D)
        solver.limiters = fortran_lim
    else:
        solver = pyclaw.ClawSolver1D(roe_efix)
        solver.kernel_language = "Python"
        solver.num_eqn, solver.num_waves, solver.fwave = 3, 3, False
        if callable(limiter) or isinstance(limiter, (str, tuple)):
            limiter = [limiter] * 3
        solver.limiters = [register(l, l) if callable(l) else register(*l) if isinstance(l, tuple) else l for l in limiter]
    bc = {"extrap": pyclaw.BC.extrap, "wall": pyclaw.BC.wall, "periodic": pyclaw.BC.periodic}[P["bc"]]
    solver.bc_lower[0] = solver.bc_upper[0] = bc
    solver.cfl_desired, solver.cfl_max = cfl, min(1.0, cfl + 0.05)
    x = pyclaw.Dimension(P["a"], P["b"], N, name="x")
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain, 3)
    state.problem_data.update(PD)
    e = np.linspace(P["a"], P["b"], N + 1)
    state.q[...] = cell_avg_cons(P["ic"], e)
    solver.dt_initial = 1e-3 * (P["b"] - P["a"]) / N
    solver.max_steps = 10 ** 7
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain); claw.solver = solver
    claw.tfinal = P["t"]; claw.num_output_times = 1; claw.keep_copy = True
    claw.output_format = None; claw.verbosity = 0
    claw.run()
    q = claw.frames[-1].q
    xc = 0.5 * (e[1:] + e[:-1])
    rho = q[0]; u = q[1] / rho; p = (GAMMA - 1) * (q[2] - 0.5 * rho * u ** 2)
    return xc, rho, u, p


# ---------------- solucion exacta del problema de Riemann (Toro, cap. 4) ----------------
def exact_riemann(L, R, x0, t, x):
    rl, ul, pl = L; rr, ur, pr = R; g = GAMMA
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def fk(p, rk, pk, ck):
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    f = lambda p: fk(p, rl, pl, cl) + fk(p, rr, pr, cr) + ur - ul
    lo, hi = 1e-10, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    ps = 0.5 * (lo + hi)
    us = 0.5 * (ul + ur) + 0.5 * (fk(ps, rr, pr, cr) - fk(ps, rl, pl, cl))
    xi = (np.asarray(x) - x0) / t
    rho = np.empty_like(xi)
    for i, s in enumerate(xi):
        if s < us:   # izquierda del contacto
            if ps > pl:
                rs = rl * (ps / pl + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pl + 1)
                S = ul - cl * np.sqrt((g + 1) / (2 * g) * ps / pl + (g - 1) / (2 * g))
                rho[i] = rl if s < S else rs
            else:
                rs = rl * (ps / pl) ** (1 / g); cs = cl * (ps / pl) ** ((g - 1) / (2 * g))
                if s < ul - cl: rho[i] = rl
                elif s > us - cs: rho[i] = rs
                else: rho[i] = rl * (2 / (g + 1) + (g - 1) / ((g + 1) * cl) * (ul - s)) ** (2 / (g - 1))
        else:
            if ps > pr:
                rs = rr * (ps / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pr + 1)
                S = ur + cr * np.sqrt((g + 1) / (2 * g) * ps / pr + (g - 1) / (2 * g))
                rho[i] = rr if s > S else rs
            else:
                rs = rr * (ps / pr) ** (1 / g); cs = cr * (ps / pr) ** ((g - 1) / (2 * g))
                if s > ur + cr: rho[i] = rr
                elif s < us + cs: rho[i] = rs
                else: rho[i] = rr * (2 / (g + 1) - (g - 1) / ((g + 1) * cr) * (ur - s)) ** (2 / (g - 1))
    return rho


def exact_density_avg(prob, N, sub=16):
    P = PROBLEMS[prob]
    LR = {"sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1)), "lax": ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571))}[prob]
    e = np.linspace(P["a"], P["b"], N + 1)
    xs = (e[:-1, None] + (np.arange(sub) + 0.5)[None, :] / sub * (e[1] - e[0])).ravel()
    return exact_riemann(LR[0], LR[1], 0.5, P["t"], xs).reshape(N, sub).mean(axis=1)


def solve_f(prob, N, specs, cfl=0.9):
    """igual que solve() pero con el kernel Fortran modificado (clawhyb) y limitadores por codigo"""
    import clawhyb
    P = PROBLEMS[prob]
    solver = pyclaw.ClawSolver1D(riemann.euler_with_efix_1D)
    solver.fmod = clawhyb.FMOD[1]
    if not isinstance(specs, list):
        specs = [specs] * 3
    solver.limiters = [clawhyb.code(s) for s in specs]
    bc = {"extrap": pyclaw.BC.extrap, "wall": pyclaw.BC.wall, "periodic": pyclaw.BC.periodic}[P["bc"]]
    solver.bc_lower[0] = solver.bc_upper[0] = bc
    solver.cfl_desired, solver.cfl_max = cfl, min(1.0, cfl + 0.05)
    solver.max_steps = 10 ** 7
    x = pyclaw.Dimension(P["a"], P["b"], N, name="x")
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain, 3)
    state.problem_data.update(PD)
    e = np.linspace(P["a"], P["b"], N + 1)
    state.q[...] = cell_avg_cons(P["ic"], e)
    solver.dt_initial = 1e-3 * (P["b"] - P["a"]) / N
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain); claw.solver = solver
    claw.tfinal = P["t"]; claw.num_output_times = 1; claw.keep_copy = True
    claw.output_format = None; claw.verbosity = 0
    claw.run()
    q = claw.frames[-1].q
    rho = q[0]; u = q[1] / rho; p = (GAMMA - 1) * (q[2] - 0.5 * rho * u ** 2)
    return 0.5 * (e[1:] + e[:-1]), rho, u, p
