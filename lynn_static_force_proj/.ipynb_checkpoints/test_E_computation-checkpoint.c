#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <qcd.h>

static const read_env_int(
    const char *name,
    const int default_value,
    const int min_value,
    const int max_value
){
    const char *text = getenv(name);
    if (text == NULL || *text == '\0'){
        return default_value;
    } 

    errno = 0;
    char *end = NULL;
    const long value = strtol(text, &end, 10);

    if(
        errno == ERANGE ||
        end == text ||
        *end != '\0' ||
        value < min_value ||
        value > max_value
    ) {
        fprintf(
            stderr, 
            "Error: %s must be an integer in [%d, %d], got '%s' \n",
            name, min_value, max_value, text
        );
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
static double su3_traceless_F_from_Q(
    qcd_complex_16 Q[3][3]
) {
    qcd_complex_16 F[3][3];
    /* 
    Computing Field Strength Tensor:
    F_munu = anti-Hermitian part of Q_munu with factor 1/8.
    then subtract the color tract to project to su(3)
    */
    for (int a = 0; a < 3; a++){
        for (int b = 0; b < 3; b++) {
            qcd_complex_16 Q_dagger = qcd_CONJ(Q[b][a]);
            F[a][b].re = 0.125 * (Q[a][b].re - Q_dagger.re);
            F[a][b].im = 0.125 * (Q[a][b].im + Q_dagger.im);
        }
    }
    // computing F_munu_trace
    qcd_complex_16 tr = {0.0, 0.0};

    for (int a = 0; a < 3; a++){
        tr.re += F[a][a].re;
        tr.im += F[a][a].im;
    }
    tr.re /= 3.0;
    tr.im /= 3.0;
    // projection onto su(3)
    for (int a = 0; a < 3; a++){
        F[a][a].re -= tr.re;
        F[a][a].im -= tr.im;      
    }
    return F;    
}
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int myid = 0 ;
    int numprocs = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs != 1) {
        if (myid == 0) {
            fprintf(stderr, "Error: this test is intended for only 1 MPI Rank. \n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int nc = read_env_int("NCfg", 1, 1, INT_MAX);

    int L[4] = {4,2,2,2}; // vector of Lattice dimension
    int P[4] = {1,1,1,1}; // vector of Process to use in each direction
    double theta[3] = {0.0, 0.0, 0.0}; // vector of angles for generalized periodic boundaries for fermion fields
    
    qcd_geometry geo; // pointer to the geometry structure that is initialized
    qcd_gaugeField u; // pointer to a gauge field structure
    qcd_cloverField clv; // pointer to a clover field structure

    if(qcd_initGeometry(&geo, L, P, theta, myid, numprocs)){
        fprintf(stderr, "Error: qcd_initGeometry failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }   
    // Check even/odd decomposition support to the geometry structure geo
    if(qcd_initEO(&geo)){
        fprintf(stderr, "Error: qcd_initEO failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }  

    if (qcd_initGaugeField(&u, &geo)) {
        fprintf(stderr, "ERROR: qcd_initGaugeField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (qcd_initCloverField(&clv, &geo)) {
        fprintf(stderr, "ERROR: qcd_initCloverField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    char gauge_field_path[1024];
    sprintf(
        gauge_field_path, 
        sizeof(gauge_field_path),
        "/home/m2427382/Wilson-Loop-Project/Em1/cnfg/Em1p4n1"
        ""
    )
    }


}