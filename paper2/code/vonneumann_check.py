"""Verificacion simbolica de la Proposicion (analisis de von Neumann cerca de estados suaves):
esquema de un paso con Psi(r) ~ 1 + beta (r - 1)."""
import sympy as sp
xi, nu, b = sp.symbols('xi nu beta', real=True)
I = sp.I
E, Ep = sp.exp(-I * xi), sp.exp(I * xi)
G = 1 - nu * (1 - E) - nu * (1 - nu) / 2 * (1 - E) * ((1 - b) * (Ep - 1) + b * (1 - E))
err = sp.expand(sp.series(G - sp.exp(-I * nu * xi), xi, 0, 5).removeO())
for k in range(5):
    print(f"coef xi^{k}:", sp.factor(sp.simplify(err.coeff(xi, k))))
print("coef xi^3 = 0  <=>  beta =", sp.solve(sp.Eq(sp.simplify(err.coeff(xi, 3)), 0), b))
G2 = sp.expand(G * sp.conjugate(G))
print("|G|^2 - 1 ~", sp.factor(sp.simplify(sp.series(G2, xi, 0, 5).removeO() - 1)))
