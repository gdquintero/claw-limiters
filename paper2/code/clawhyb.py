"""Modulos Fortran de CLAWPACK clasico (1D y 2D) con limitadores dependientes del CFL local
(limiter.f90 modificado para pasar |s| dt/dx a philim) y los limitadores nuevos."""
import sys, os, importlib
_here = os.path.dirname(os.path.abspath(__file__))
for d in ("fortran/build1", "fortran/build2"):
    p = os.path.join(_here, d)
    if p not in sys.path:
        sys.path.insert(0, p)
import classic1hyb, classic2hyb

FMOD = {1: classic1hyb, 2: classic2hyb}
_slots = {}


def slot(delta, theta):
    """devuelve k (1..20) con hdelta(k)=delta, htheta(k)=theta en ambos modulos"""
    key = (float(delta), float(theta))
    if key not in _slots:
        k = len(_slots) + 1
        assert k <= 20
        for m in FMOD.values():
            m.limpar.hdelta[k - 1] = key[0]
            m.limpar.htheta[k - 1] = key[1]
        _slots[key] = k
    return _slots[key]


def code(spec):
    """spec: 'MC','superbee','minmod','vanleer','minbee','hyb0','beta',
       ('AR',theta), ('CS',theta), ('H',delta,theta)"""
    fixed = {"minmod": 1, "superbee": 2, "vanleer": 3, "MC": 4, "minbee": 21, "hyb0": 22, "beta": 401}
    if isinstance(spec, str):
        return fixed[spec]
    kind = spec[0]
    if kind == "AR": return 200 + slot(0.0, spec[1])
    if kind == "CS": return 300 + slot(0.0, spec[1])
    if kind == "H": return 100 + slot(spec[1], spec[2])
    raise ValueError(spec)
