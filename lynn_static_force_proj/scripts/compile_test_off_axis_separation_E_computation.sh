#!/bin/bash
set -euo pipefail

cd /home/m2427382/Wilson-Loop-Project/lynn_static_force_proj

module purge
module load compiler/gcc/10.2.1
module load mpi/openmpi/4.1.0-no_ucx
module load mkl/latest

HOMEBASE=/home/m2427382/Wilson-Loop-Project
MKLROOT_LOCAL=${MKLROOT}

QCDINC=${HOMEBASE}/qcd/include
QCDLIB=${HOMEBASE}/qcd/lib

CFLAGS="-O2 -Wall -I${QCDINC}"
LDFLAGS="-L${QCDLIB} -Wl,-rpath,${QCDLIB} -lqcd"
MKLLDFLAGS="-L${MKLROOT_LOCAL}/lib/intel64 -Wl,-rpath,${MKLROOT_LOCAL}/lib/intel64 -lmkl_rt -lpthread -ldl"

mkdir -p build

rm -f build/test_off_axis_separation_E_computation build/test_off_axis_separation_E_computation.o

mpicc ${CFLAGS} -c src/test_off_axis_separation_E_computation.c -o build/test_off_axis_separation_E_computation.o

mpicxx -o build/test_off_axis_separation_E_computation \
  build/test_off_axis_separation_E_computation.o \
  ${LDFLAGS} \
  ${MKLLDFLAGS} \
  -lz -lm

echo "Built build/test_off_axis_separation_E_computation"
ldd build/test_off_axis_separation_E_computation | egrep "not found|mpi|mkl|qcd|z" || true
