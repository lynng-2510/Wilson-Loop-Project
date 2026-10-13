#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <qcd.h>

/* Computing local Gauge-field matrix Ei(t,x,y,z) */

static long site4_index(
    const int t,
    const int x,
    const int y,
    const int z,
    const int L[4]
) {
    return (((long)t * L[1] + x) * L[2] + y) * L[3] + z;
}
static int read_env_int(
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
    return (int)value; // convert long to int
}

static void su3_traceless_F_from_Q(
    const qcd_complex_16 Q[3][3], qcd_complex_16 F[3][3]
) {
    /* 
    Computing Field Strength Tensor:
    F_munu = anti-Hermitian part of Q_munu with factor 1/8.
    then subtract the color tract to project to su(3)
    */
    for (int a = 0; a < 3; a++){
        for (int b = 0; b < 3; b++) {
            F[a][b].re = 0.125 * (Q[a][b].re - Q[b][a].re);
            F[a][b].im = 0.125 * (Q[a][b].im + Q[b][a].im);
        }
    }
    // Computing F_munu_trace
    qcd_complex_16 tr = {0.0, 0.0};

    for (int a = 0; a < 3; a++){
        tr.re += F[a][a].re;
        tr.im += F[a][a].im;
    }
    tr.re /= 3.0;
    tr.im /= 3.0;
    // Projection onto su(3)
    for (int a = 0; a < 3; a++){
        F[a][a].re -= tr.re;
        F[a][a].im -= tr.im;      
    }
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

    const int nc = read_env_int("NCFG", 1, 1, INT_MAX);

    int L[4] = {48,24,24,24}; // vector of Lattice dimension
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
    snprintf(
        gauge_field_path, 
        sizeof(gauge_field_path),
        "/home/m2427382/Wilson-Loop-Project/Em1/cnfg/Em1p4n%d", nc);
    
    if (myid==0){
        printf("Clover E one-config test\n");
        printf("HOST=%s\n", getenv("HOSTNAME") ? getenv("HOSTNAME") : "(unknown)");
        printf("N-Configuration=%d\n", nc);
        printf("Gauge field: %s\n", gauge_field_path);
        fflush(stdout);
    }

    if (qcd_getGaugeField(gauge_field_path, qcd_GF_OPENQCD, &u)) {
        fprintf(stderr, "ERROR: qcd_getGaugeField failed for %s\n", gauge_field_path);
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const double plaq = qcd_calculatePlaquette(&u);

    if (myid == 0){
        printf("Average plaquette = %.16e\n", plaq);
        fflush(stdout);
    }
    if (qcd_calculateCloverField(&clv, &u)){
        fprintf(stderr, "ERROR: qcd_calculateCloverField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    /*
      qcd_cloverField plane order:
        0: 0/1
        1: 0/2
        2: 0/3
        3: 1/2
        4: 1/3
        5: 2/3

      Here 0 is Euclidean time.
      So E-like planes are 0,1,2 and B-like planes are 3,4,5.
    */

   qcd_complex_16 E[3][3][3];
    // To store E-matrices later:
    // qcd_complex_16 (*E)[3][3][3] = malloc((size_t)geo.lV * sizeof *E);
    // if (E == NULL){
    // fprintf(stderr, "ERROR: allocation for E failed");
    //  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    // }
    // for (long site4 = 0; site4 < geo.lV; site4 ++){
    //     for (int p = 0; p < 3; p++){
    //         su3_traceless_F_from_Q(clv.D[site4][p], E[site4][p]);)
    //     }
    // }
    // free(E);

    // Test with a specific site4
    const int t = 0;
    const int x = 0;
    const int y = 0;
    const int z = 0;
    const long site4 = site4_index(t, x, y, z, L);

    for (int p = 0; p < 3; p++){
        su3_traceless_F_from_Q(clv.D[site4][p], E[p]);
    }
    if (myid == 0) {
        for (int p = 0; p < 3; p++){
            printf("Computed Chromoelectric field E[%d] at site %ld:\n", p, site4);
            for (int a = 0; a < 3; a++){
                for (int b = 0; b < 3; b++) {
                    printf("E[%d][%d][%d] = %.16e %.16ei\n", 
                        p, a, b,
                        E[p][a][b].re, E[p][a][b].im);
                }
            }
        }
    }
    qcd_destroyCloverField(&clv);
    qcd_destroyGaugeField(&u);
    qcd_destroyGeometry(&geo);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

