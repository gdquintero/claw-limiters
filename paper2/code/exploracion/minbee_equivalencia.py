"""Comprobacion numerica: MINBEE = 1/2 (Superbee + Minmod) = minmod generalizado con theta=1.5
(Nessyahu-Tadmor) = MC con min{2,2r} multiplicado por theta=3/4 (Kemm 2011, p. 414)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from ideas_check import minbee, superbee, minmod
r = np.linspace(-5, 50, 2000001)
gm = lambda r, t: np.maximum(0, np.minimum(np.minimum(t * r, (1 + r) / 2), t))
kemm = lambda r, th: np.maximum(0, np.minimum(np.minimum(th * 2 * r, (1 + r) / 2), th * 2))
print("max|MINBEE - (SB+MM)/2|        =", np.abs(minbee(r) - (superbee(r) + minmod(r)) / 2).max())
print("max|MINBEE - minmod-theta 1.5| =", np.abs(minbee(r) - gm(r, 1.5)).max())
print("max|MINBEE - Kemm MC theta=3/4|=", np.abs(minbee(r) - kemm(r, 0.75)).max())
