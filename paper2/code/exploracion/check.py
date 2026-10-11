import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # modulos de ../
import sympy as sp
import numpy as np

x, b, r, h = sp.symbols('x beta r h', real=True)

# ---------- TDPUS-C3 (thesis eq. 2.54/2.56) ----------
a = {4: b, 5: 320-8*b, 6: 26*b-1664, 7: 3456-44*b, 8: 41*b-3584, 9: 1856-20*b, 10: 4*b-384}
P = x + sum(c*x**k for k, c in a.items())
conds = {
    "P(0)=0": P.subs(x, 0), "P(1)=1": sp.simplify(P.subs(x, 1)), "P(1/2)=3/4": sp.simplify(P.subs(x, sp.Rational(1, 2))),
    "P'(1/2)=3/4": sp.simplify(sp.diff(P, x).subs(x, sp.Rational(1, 2))),
    "P'(0)=1": sp.diff(P, x).subs(x, 0), "P'(1)=1": sp.simplify(sp.diff(P, x).subs(x, 1)),
    "P''(0)=0": sp.diff(P, x, 2).subs(x, 0), "P''(1)=0": sp.simplify(sp.diff(P, x, 2).subs(x, 1)),
    "P'''(0)=0": sp.diff(P, x, 3).subs(x, 0), "P'''(1)=0": sp.simplify(sp.diff(P, x, 3).subs(x, 1)),
}
print("TDPUS-C3 conditions:", {k: sp.nsimplify(v) for k, v in conds.items()})

# limiter: phi_f = phi_U + 1/2 Psi(r) (1-phi_U), r = phi_U/(1-phi_U)  => phi_U = r/(1+r)
xr = r/(1+r)
Psi = sp.simplify(2*(P.subs(x, xr) - xr)/(1 - xr))
Psi_thesis = ((2*b-128)*r**6 + (640-4*b)*r**5 + 2*b*r**4)/(1+r)**9
print("Psi(r) =", sp.factor(Psi))
print("thesis eq 2.57 matches:", sp.simplify(Psi - Psi_thesis) == 0)
print("Psi(1) =", sp.simplify(Psi.subs(r, 1)), " Psi'(1) =", sp.simplify(sp.diff(Psi, r).subs(r, 1)))
print("Psi ~ r->0:", sp.series(Psi, r, 0, 6).removeO(), " ; r->oo leading:", sp.limit(Psi*r**3, r, sp.oo))

bv = 567.25
Pn = sp.lambdify(x, P.subs(b, bv), 'numpy'); Psin = sp.lambdify(r, Psi.subs(b, bv), 'numpy')
xs = np.linspace(0, 1, 200001); rs = np.linspace(1e-6, 60, 600001)
print("beta=567.25: CBC  min(P-x)=%.3e  max(P-1)=%.3e" % ((Pn(xs)-xs).min(), (Pn(xs)-1).max()))
print("beta=567.25: TVD(NV) max(P-2x)=%.3e" % ((Pn(xs)-2*xs).max()))
print("beta=567.25: max Psi=%.4f at r=%.3f ; max Psi/r=%.4f ; Psi(10)=%.3f Psi(2)=%.3f Psi(0.5)=%.3f" % (
    Psin(rs).max(), rs[Psin(rs).argmax()], (Psin(rs)/rs).max(), Psin(10.0), Psin(2.0), Psin(0.5)))

# ---------- semi-discrete order of face-interpolation schemes (point values, conservative FD) ----------
def deriv_order(fface):
    # fface: face value at i+1/2 as function of (q_{i-1}, q_i, q_{i+1}); derivative D = (F_{i+1/2}-F_{i-1/2})/h
    xx = sp.symbols('xx')
    q = sp.Function('q')
    Fp = fface(q(xx-h), q(xx), q(xx+h)); Fm = fface(q(xx-2*h), q(xx-h), q(xx))
    D = (Fp - Fm)/h
    s = sp.series(D, h, 0, 4).removeO().doit()
    return sp.simplify(s - sp.diff(q(xx), xx))

kap = sp.symbols('kappa')
kface = lambda R, U, D: U + (1+kap)/4*(D-U) + (1-kap)/4*(U-R)
err = deriv_order(kface)
print("kappa-scheme derivative error:", sp.factor(sp.simplify(err)))

# ---------- Psi'(1) and the kappa / NVD slope relation ----------
psi_k = (1+kap)/2 + (1-kap)/2*r
print("kappa: Psi'(1) =", sp.diff(psi_k, r), " NVD slope at Q =", sp.simplify(1 - kap/2))
