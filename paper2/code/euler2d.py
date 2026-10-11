"""Paso 4: Euler 2D en CLAWPACK clasico (PyClaw, Fortran, ondas transversales) con limitadores
dependientes del CFL local (modulo clawhyb). Ondas de euler_4wave_2D: [u-c, entropia, cizalla, u+c]."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from clawpack import pyclaw, riemann
import clawhyb

GAMMA = 1.4


def prim2cons(rho, u, v, p):
    return np.array([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u ** 2 + v ** 2)])


def cell_avg2(prim, xe, ye, m=4):
    g, w = np.polynomial.legendre.leggauss(m)
    xc, xr = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (xe[1:] - xe[:-1])
    yc, yr = 0.5 * (ye[1:] + ye[:-1]), 0.5 * (ye[1:] - ye[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    out = 0
    for gi, wi in zip(g, w):
        for gj, wj in zip(g, w):
            out = out + wi * wj * prim2cons(*prim(X + xr[0] * gi, Y + yr[0] * gj))
    return out / 4


# ---------------- problemas ----------------
def quadrants(x, y):          # Liska-Wendroff caso 3 = Kemm 5.3 (Schulz-Rinne)
    r, t = x >= 0.8, y >= 0.8
    l, b = ~r, ~t
    rho = 1.5 * r * t + 0.532258064516129 * l * t + 0.137992831541219 * l * b + 0.532258064516129 * r * b
    u = 1.206045378311055 * l
    v = 1.206045378311055 * b
    p = 1.5 * r * t + 0.3 * l * t + 0.029032258064516 * l * b + 0.3 * r * b
    return rho, u, v, p


def implosion(x, y):          # Toro / Kemm 5.4, cuarto de dominio
    inside = x ** 2 + y ** 2 < 0.4 ** 2
    rho = np.where(inside, 0.125, 1.0); p = np.where(inside, 0.1, 1.0)
    return rho, 0 * x, 0 * x, p


GRESHO_U0, GRESHO_C = 1.0, (0.5, 0.5)


def gresho_uphi_p(r):
    uphi = np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))
    rs = np.maximum(r, 1e-300)
    p = np.where(r < 0.2, 5 + 12.5 * r ** 2,
                 np.where(r < 0.4, 9 - 4 * np.log(0.2) + 12.5 * r ** 2 - 20 * r + 4 * np.log(rs), 3 + 4 * np.log(2)))
    return uphi, p


def gresho(x, y, t=0.0):      # vortice de Gresho trasladado con velocidad (U0, 0), periodico en [0,4]x[0,1]
    xc = (GRESHO_C[0] + GRESHO_U0 * t) % 4.0
    dx = (x - xc + 2.0) % 4.0 - 2.0
    dy = y - GRESHO_C[1]
    r = np.sqrt(dx ** 2 + dy ** 2)
    uphi, p = gresho_uphi_p(r)
    rs = np.maximum(r, 1e-300)
    return np.ones_like(x), GRESHO_U0 - uphi * dy / rs, uphi * dx / rs, p


PROB2 = {
    "quadrants": dict(lo=(0, 0), hi=(1, 1), t=0.8, ic=quadrants, bc=("extrap", "extrap", "extrap", "extrap"), aspect=1),
    "implosion": dict(lo=(0, 0), hi=(1, 1), t=0.25, ic=implosion, bc=("wall", "extrap", "wall", "extrap"), aspect=1),
    "gresho": dict(lo=(0, 0), hi=(4, 1), t=3.0, ic=gresho, bc=("periodic",) * 4, aspect=4),
}
BCS = {"extrap": pyclaw.BC.extrap, "wall": pyclaw.BC.wall, "periodic": pyclaw.BC.periodic}


def run2d(prob, N, specs, cfl=0.9, fortran_default=None):
    """N = celdas en y (en x: aspect*N). specs: lista de 4 limitadores (ver clawhyb.code) o uno solo."""
    P = PROB2[prob]
    solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
    solver.transverse_waves = 2
    solver.dimensional_split = False
    if fortran_default is not None:          # modulo original de PyClaw (para validar)
        solver.limiters = fortran_default
    else:
        solver.fmod = clawhyb.FMOD[2]
        if not isinstance(specs, list):
            specs = [specs] * 4
        solver.limiters = [clawhyb.code(s) for s in specs]
    bl, bu, bb, bt = (BCS[b] for b in P["bc"])
    solver.bc_lower[0], solver.bc_upper[0], solver.bc_lower[1], solver.bc_upper[1] = bl, bu, bb, bt
    solver.cfl_desired, solver.cfl_max = cfl, min(1.0, cfl + 0.1)
    solver.max_steps = 10 ** 7
    nx, ny = P["aspect"] * N, N
    domain = pyclaw.Domain(list(P["lo"]), list(P["hi"]), [nx, ny])
    state = pyclaw.State(domain, 4)
    state.problem_data["gamma"] = GAMMA
    xe = np.linspace(P["lo"][0], P["hi"][0], nx + 1); ye = np.linspace(P["lo"][1], P["hi"][1], ny + 1)
    state.q[...] = cell_avg2(P["ic"], xe, ye)
    solver.dt_initial = 1e-3 / N
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain); claw.solver = solver
    claw.tfinal = P["t"]; claw.num_output_times = 1; claw.keep_copy = True
    claw.output_format = None; claw.verbosity = 0
    claw.run()
    q = claw.frames[-1].q.copy()
    X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), indexing="ij")
    return X, Y, q


def prims(q):
    rho = q[0]; u = q[1] / rho; v = q[2] / rho
    return rho, u, v, (GAMMA - 1) * (q[3] - 0.5 * rho * (u ** 2 + v ** 2))
