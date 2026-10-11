import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
import numpy as np
from limlib import *
from advtests import run
from ideas_tests import cell_avg
C = {"AR": arora_roe, "beta=2/3": beta_lim, "CFL-SB": cfl_superbee,
     "RES .05": lambda r, nu: resblend(r, nu, 0.05), "RES .1": lambda r, nu: resblend(r, nu, 0.1),
     "RES .2": lambda r, nu: resblend(r, nu, 0.2)}
Ns = [100, 200, 400, 800, 1600, 3200, 6400]
for nu in (0.5,):
    print(f"sin(6 pi x), T=2, nu={nu}: L1 error (observed order)")
    for n, f in C.items():
        errs = []
        for N in Ns:
            e = np.linspace(-1, 1, N + 1); q0 = cell_avg(lambda x: np.sin(6*np.pi*x), e)
            q = run(q0, nu, int(round(2/(nu*2/N))), f); errs.append(np.abs(q-q0).sum()*2/N)
        print(f"{n:9s}" + "".join(f" {errs[0]:.1e}" if k == 0 else f" {errs[k]:.1e}({np.log2(errs[k-1]/errs[k]):.1f})" for k in range(len(Ns))))
