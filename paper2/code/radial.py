"""Referencia 1D radial (cilindrica) para la implosion: Clawpack 1D Fortran + termino fuente geometrico (Strang)."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from clawpack import pyclaw, riemann
GAMMA = 1.4


def source(solver, state, dt):
    q = state.q; r = state.grid.x.centers
    def rhs(q):
        rho, mom, E = q; u = mom / rho; p = (GAMMA - 1) * (E - 0.5 * rho * u ** 2)
        return -np.array([mom, mom * u, u * (E + p)]) / r
    q1 = q + dt * rhs(q)                       # RK2 (Heun)
    q[...] = 0.5 * (q + q1 + dt * rhs(q1))


def implosion_radial(N=8000, R=1.5, t=0.25, lim=4):
    solver = pyclaw.ClawSolver1D(riemann.euler_with_efix_1D)
    solver.limiters = lim
    solver.step_source = source; solver.source_split = 2
    solver.bc_lower[0] = pyclaw.BC.wall; solver.bc_upper[0] = pyclaw.BC.extrap
    solver.cfl_desired, solver.cfl_max = 0.8, 0.9; solver.max_steps = 10 ** 7
    x = pyclaw.Dimension(0.0, R, N, name="x"); domain = pyclaw.Domain([x])
    state = pyclaw.State(domain, 3); state.problem_data.update({"gamma": GAMMA, "gamma1": GAMMA - 1, "efix": True})
    rc = x.centers
    rho = np.where(rc < 0.4, 0.125, 1.0); p = np.where(rc < 0.4, 0.1, 1.0)
    state.q[0] = rho; state.q[1] = 0; state.q[2] = p / (GAMMA - 1)
    solver.dt_initial = 1e-5
    claw = pyclaw.Controller(); claw.solution = pyclaw.Solution(state, domain); claw.solver = solver
    claw.tfinal = t; claw.num_output_times = 1; claw.keep_copy = True; claw.output_format = None; claw.verbosity = 0
    claw.run()
    return rc, claw.frames[-1].q[0].copy()


if __name__ == "__main__":
    r4, d4 = implosion_radial(4000)
    r8, d8 = implosion_radial(16000)
    i = np.interp(r4, r8, d8)
    m = r4 < 1.0
    print("diferencia L1 entre referencias 4000 y 16000 (r<1):", np.abs(d4[m] - i[m]).mean(), " rho[min,max]", d8.min(), d8.max())
    np.save("implosion_ref.npy", np.array([r8, d8]))
