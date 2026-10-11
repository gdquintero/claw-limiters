"""Limiter library for the one-step (CLAWPACK / wave-propagation) framework.
All functions: psi(r, nu) with r = upwind jump / local jump (Kemm r = LeVeque theta),
nu = local Courant number in (0,1). Formulas follow Kemm (2011) equation numbers.
"""
import numpy as np

EPS_NU = 1e-3


def _nu(nu):
    return np.clip(nu, EPS_NU, 1 - EPS_NU)


def phi3(r, nu):                                  # eq. (10): Roe's third-order linear scheme
    return 1 + (1 + nu) / 3 * (r - 1)


def ub(r, nu, theta=1.0):                         # eq. (22): upper bound of general TVD region
    nu = _nu(nu)
    return np.maximum(0, np.minimum(theta * 2 * r / nu, theta * 2 / (1 - nu)))


# ---------------- CFL-independent classics ----------------
def minmod(r, nu=None):   return np.maximum(0, np.minimum(1, r))
def superbee(r, nu=None): return np.maximum(0, np.maximum(np.minimum(1, 2 * r), np.minimum(r, 2)))
def mc(r, nu=None):       return np.maximum(0, np.minimum(np.minimum(2 * r, (1 + r) / 2), 2))
def vanleer(r, nu=None):  return (r + np.abs(r)) / (1 + np.abs(r))


# ---------------- Kemm CFL-dependent family ----------------
def arora_roe(r, nu, theta=1.0):                  # eq. (32) (theta: van Leer scaling of ub, no negative part)
    return np.maximum(0, np.minimum(ub(r, nu, theta), phi3(r, nu)))


def theta_ar(r, nu, theta=0.75):                  # eq. (40): theta-version allowing negative values
    nu = _nu(nu)
    a = np.maximum(-(1 - theta) * 2 / nu, phi3(r, nu))
    b = np.maximum(-(1 - theta) * 2 / (1 - nu) * r, theta * 2 / nu * r)
    return np.minimum(np.minimum(a, b), theta * 2 / (1 - nu))


def cfl_superbee(r, nu, theta=1.0):               # eq. (41): Roe's Ultrabee
    return np.minimum(ub(r, nu, theta), np.maximum(1, r)) * (r > 0)


def beta_lim(r, nu, beta=2 / 3):                  # eq. (43)
    s = (1 + nu) / 3
    m = np.maximum(1 + (s - beta / 2) * (r - 1), 1 + (s + beta / 2) * (r - 1))
    return np.maximum(0, np.minimum(ub(r, nu), m))


def harten_eta(r):                                # eq. (44)
    return np.abs(1 - r) / (1 + np.abs(r))


def beta_switch(r, nu):                           # eq. (45): beta = 1/3 + 2/3 eta
    return beta_lim(r, nu, beta=1 / 3 + 2 / 3 * harten_eta(r))


def hyperbee(r, nu):                              # eq. (46)
    nu = _nu(nu)
    out = np.zeros_like(r, dtype=float)
    pos = r > 0
    rp = r[pos] if np.ndim(r) else r
    nup = nu[pos] if np.ndim(nu) else nu
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 2 * rp / (nup * (1 - nup)) * (nup * (rp - 1) + 1 - rp ** nup) / (rp - 1) ** 2
    near1 = np.abs(rp - 1) < 1e-4
    # Taylor value at r=1 is 1 (and slope phi3'); use phi3 there to avoid cancellation
    val = np.where(near1, phi3(rp, nup), val)
    out[pos] = val
    return out


def superpower(r, nu):                            # eqs. (57)-(58)
    nu = _nu(nu)
    s = (1 + nu) / 3
    p = np.where(r <= 1, 2 / nu * 2 * (1 - s), 2 / (1 - nu) * 2 * s)
    ra = np.abs(r)
    return np.maximum(0, phi3(r, nu) * (1 - np.abs((1 - ra) / (1 + ra)) ** p)) * (r > 0)


def hyperpower(r, nu):                            # eq. (59)
    return np.maximum(superpower(r, nu), hyperbee(r, nu))


# ---------------- NEW: third-order-anchored compressive blend ----------------
def w_harten(r, q=1.0):
    return harten_eta(r) ** q


def w_user(r):                                    # complement of the user's idea-2 sensor: ((1-r^2)/(1+r^2))^2
    return ((1 - r ** 2) / (1 + r ** 2)) ** 2


def blend(r, nu, lam=1.0, w=w_harten, theta=1.0):
    """psi = AR + lam*w(r)*(CS - AR).  TVD (general) + inclusion (M2) by convexity,
    Roe third-order tangency at r=1 for any w with w(1)=0 (since CS-AR = O(r-1))."""
    a = arora_roe(r, nu, theta)
    c = cfl_superbee(r, nu, theta)
    return a + lam * w(r) * (c - a)


LIMITERS = {
    "minmod": minmod, "vanLeer": vanleer, "MC": mc, "superbee": superbee,
    "AR": arora_roe, "theta-AR(.75)": theta_ar, "CFL-SB": cfl_superbee,
    "beta=2/3": beta_lim, "beta-switch": beta_switch,
    "Hyperbee": hyperbee, "Superpower": superpower, "Hyperpower": hyperpower,
}


def superpower_fixed(r, nu):
    """Superpower with exponents chosen so the TVD bound is met sharply for every nu:
    slope at r->0 = 2p(1-phi3') = 2/nu  and  limit r->inf = 2p*phi3' = 2/(1-nu).
    (Kemm eq. 58 as printed multiplies by 2(1-phi3'), 2phi3'; that coincides only at nu=1/2.)"""
    nu = _nu(nu)
    s = (1 + nu) / 3
    p = np.where(r <= 1, 1 / (nu * (1 - s)), 1 / ((1 - nu) * s))
    ra = np.abs(r)
    return np.maximum(0, phi3(r, nu) * (1 - np.abs((1 - ra) / (1 + ra)) ** p)) * (r > 0)


LIMITERS["Superpower-fixed"] = superpower_fixed


def w_res(r, eta0=0.1, kind="rational"):
    """Resolution-adaptive weight: ~0 where r~1 (well resolved, keeps 3rd-order tangency, w=O((r-1)^2)),
    ~1 once the Harten deviation eta=|1-r|/(1+|r|) exceeds the threshold eta0 (under-resolved / discontinuous)."""
    x = harten_eta(r) / eta0
    return x ** 2 / (1 + x ** 2) if kind == "rational" else 1 - np.exp(-x ** 2)


def resblend(r, nu, eta0=0.1, lam=1.0, theta=1.0):
    return blend(r, nu, lam, lambda rr: w_res(rr, eta0), theta)
