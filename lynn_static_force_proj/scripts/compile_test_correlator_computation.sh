#!/bin/bash
set -euo pipefail

cd /home/m2427382/Wilson-Loop-Project/lynn_static_force_proj

module purge
module load compiler/gcc/10.2.1
module load mpi/openmpi/4.1.0-no_ucx
module load mkl/latest

HOMEBASE=/home/m2427382/Wilson-Loop-Project
H5ROOT=${HOMEBASE}/extern/hdf5-1.14.6-openmpi410
MKLROOT_LOCAL=${MKLROOT}

QCDINC=${HOMEBASE}/qcd/include
QCDLIB=${HOMEBASE}/qcd/lib

EVECH5=${HOMEBASE}/disteigvecshdf5/c

CFLAGS="-O2 -Wall -I${QCDINC} -I${EVECH5}/include -I${H5ROOT}/include"
LDFLAGS="-L${QCDLIB} -Wl,-rpath,${QCDLIB} -lqcd"
H5LDFLAGS="-L${H5ROOT}/lib -Wl,-rpath,${H5ROOT}/lib -lhdf5_hl -lhdf5"
MKLLDFLAGS="-L${MKLROOT_LOCAL}/lib/intel64 -Wl,-rpath,${MKLROOT_LOCAL}/lib/intel64 -lmkl_rt -lpthread -ldl"

mkdir -p build

rm -f build/test_correlator_computation build/test_correlator_computation.o

mpicc ${CFLAGS} -c ${EVECH5}/src/DistEigvecsHdf5Reader.c -o build/DistEigvecsHdf5Reader_RT.o
mpicc ${CFLAGS} -c src/test_correlator_computation.c -o build/test_correlator_computation.o

mpicxx -o build/test_correlator_computation \
  build/test_correlator_computation.o \
  build/DistEigvecsHdf5Reader_RT.o \
  ${LDFLAGS} \
  ${H5LDFLAGS} \
  ${MKLLDFLAGS} \
  -lz -lm

echo "Built build/test_correlator_computation"
ldd build/test_correlator_computation | egrep "not found|mpi|mkl|qcd|z" || true
