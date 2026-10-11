# Código del paper 2 (TOPBEE-δ)

Programas que generan todos los resultados, las figuras (`../figures/`) y las tablas (`../tables.tex`) del borrador `../main.tex`.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

También hace falta `gfortran` para compilar el kernel de CLAWPACK modificado.

## Regenerar todo (unos 20 minutos)

```bash
./run_all.sh
```

## Qué hace cada archivo

| Archivo | Contenido |
|---|---|
| `ideas_check.py` | MINBEE, TOPUS, sensor ω e híbrido original (ideas 1 y 2) |
| `limlib.py` | limitadores de referencia dependientes del CFL (Kemm 2011): Arora–Roe, CFL-Superbee, β, ... |
| `limprops.py` | verificación de propiedades: TVD, inclusión (M2), monotonía (M5), tangencia de tercer orden |
| `ideas_tests.py`, `advtests.py` | esquema de un paso para advección lineal, datos iniciales (Zalesak) |
| `hyb1.py`, `hyb2.py` | familia TOPBEE-δ y pruebas de advección lineal (Tabla 1, Fig. 3) |
| `vonneumann_check.py` | verificación simbólica del análisis de von Neumann (Sección 5) |
| `euler1d.py` | Euler 1D en PyClaw: solver de Roe con corrección de entropía, problemas, solución exacta |
| `euler_tests.py`, `euler_tests2.py`, `euler_pareto.py` | pruebas de Euler 1D (Tabla 2, Fig. 4) |
| `fortran/` | kernel clásico de CLAWPACK modificado: el limitador recibe el número de Courant local de cada onda (`limiter.f90`) y contiene los limitadores nuevos (`philim.f90`) |
| `build_fortran.sh`, `clawhyb.py` | compilación del kernel y su uso desde Python |
| `euler2d.py`, `euler2d_tests.py`, `radial.py` | Euler 2D: problema de Riemann, implosión, vórtice de Gresho (Tabla 3, Fig. 5) |
| `paper_figs.py`, `paper_tables.py` | figuras y tablas del borrador |

Los archivos de `fortran/` derivan de Clawpack 5 (`pyclaw/src/pyclaw/classic`, licencia BSD de 3 cláusulas, ver `fortran/LICENSE_CLAWPACK`).
