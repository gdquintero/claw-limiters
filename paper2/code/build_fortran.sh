#!/bin/bash
# Compila los kernels clasicos de CLAWPACK modificados (limitador con numero de Courant local por onda).
# Requiere gfortran y, en el PATH, el python del entorno con numpy, meson y ninja.
set -e
cd "$(dirname "$0")/fortran"
rm -rf build1 build2 && mkdir build1 build2
cp limiter.f90 philim.f90 step1.f90 build1/
cp limiter.f90 philim.f90 flux2.f90 step2ds.f90 step2.f90 build2/
(cd build1 && python -m numpy.f2py -c -m classic1hyb limiter.f90 philim.f90 step1.f90 > build.log 2>&1)
(cd build2 && python -m numpy.f2py -c -m classic2hyb limiter.f90 philim.f90 flux2.f90 step2ds.f90 step2.f90 > build.log 2>&1)
ls build1/*.so build2/*.so
