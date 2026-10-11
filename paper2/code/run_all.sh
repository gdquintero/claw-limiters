#!/bin/bash
# Regenera todos los datos, las figuras (../figures/) y las tablas (../tables.tex) del borrador.
# Tiempo aproximado: 20 minutos. Ejecutar con el entorno de Python activado.
set -e
cd "$(dirname "$0")"
./build_fortran.sh
python vonneumann_check.py     # Proposicion de la seccion 5 (expansion de von Neumann)
python hyb1.py                 # adveccion lineal: hibrido original y primeras variantes -> hyb1_conv.npy
python hyb2.py                 # adveccion lineal: familia TOPBEE-delta             -> hyb2_res.npy
python euler_pareto.py         # Euler 1D, barrido de delta en el contacto          -> euler_pareto.npy
python radial.py               # referencia radial 1D de la implosion               -> implosion_ref.npy
python euler2d_tests.py        # Euler 2D (incluye referencias 800x800, ~11 min)    -> euler2d_res.npy, quad_*.npy
python paper_figs.py           # figuras  -> ../figures/fig1..fig5
python paper_tables.py > /dev/null   # tablas -> ../tables.tex
