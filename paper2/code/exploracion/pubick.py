import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
import numpy as np
def pubick(x, m1, m2):
    x = np.asarray(x, float); out = x.copy()
    a = (x >= 0) & (x < 0.5); c = (x >= 0.5) & (x <= 1)
    xa = x[a]; xc = x[c]
    out[a] = 3/(4*(4*m1-1)**2)*((4*m1)*(4*m1-1)*xa + (2*m1-1)*(2*m1-np.sqrt(4*m1**2+2*(1-4*m1)*xa)))
    out[c] = -1/(4*(4*m2-3)**2)*((4*m2-3)*(5-6*m2+4*(2-3*m2)*xc) + (1-2*m2)*(1-2*m2+np.sqrt(4*m2**2-2+2*(3-4*m2)*xc)))
    return out
def bez(t, P0, P1, P2):
    return ((1-t)**2)[:,None]*P0 + (2*t*(1-t))[:,None]*P1 + (t**2)[:,None]*P2
for m1, m2 in [(0.3, 5/6), (0.4, 0.7), (0.45, 0.55)]:
    n1 = 3*(2*m1+1)/8; n2 = 3*(2*m2+1)/8
    t = np.linspace(0, 1, 20001)
    C1 = bez(t, np.array([0,0]), np.array([m1,n1]), np.array([0.5,0.75]))
    C2 = bez(t, np.array([0.5,0.75]), np.array([m2,n2]), np.array([1,1]))
    err = max(np.abs(pubick(C1[:,0], m1, m2)-C1[:,1]).max(), np.abs(pubick(C2[:,0], m1, m2)-C2[:,1]).max())
    xs = np.linspace(0, 1, 200001); f = pubick(xs, m1, m2)
    r = xs[1:-1]/(1-xs[1:-1]); psi = 2*(f[1:-1]-xs[1:-1])/(1-xs[1:-1])
    i1 = np.argmin(np.abs(r-1)); dpsi = (psi[i1+50]-psi[i1-50])/(r[i1+50]-r[i1-50])
    print(f"mu1={m1:.3f} mu2={m2:.3f} formula-vs-Bezier err={err:.1e} | slope@O={n1/m1:.3f} slope@P={(1-n2)/(1-m2):.3f}"
          f" | CBC ok={np.all(f>=xs-1e-12) and np.all(f<=1+1e-12)} TVD(NV) ok={np.all(f<=2*xs+1e-9)}"
          f" | Psi'(1)~{dpsi:.4f} max Psi={psi.max():.3f} Psi(r->inf)~{psi[-1]:.3f} max Psi/r={(psi/r).max():.3f}")
