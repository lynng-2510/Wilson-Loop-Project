#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <qcd.h>
#include <math.h>

/* Computing local Gauge-field matrix Ei(t,x) (where i=(x,y,z)) for an off-axis separated quark-antiquark pair */

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

static int wrap_mod(const int a, const int n){
    int r = a%n;
    return (r<0) ? r + n : r;
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

    int L[4] = {48,24,24,24};
    int P[4] = {1,1,1,1};
    double theta[3] = {0.0, 0.0, 0.0}; 
    qcd_geometry geo;
    qcd_gaugeField u;
    qcd_cloverField clv;

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
    
    if (myid == 0){
        printf("Clover E one-config test:\n");
        printf("HOST=%s\n", getenv("HOSTNAME") ? getenv("HOSTNAME") : "(unknown)");
        printf("Configuration number=%d\n", nc);
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

    // Define coordinates for quark (0, 0, 0) and antiquark (rx, ry, rz)

    const int quark_x = 0;
    const int quark_y = 0;
    const int quark_z = 0;

    const int antiquark_x = 2;
    const int antiquark_y = 2;
    const int antiquark_z = 0;

    const int probe_t = 2;
    const int probe_x = quark_x;
    const int probe_y = quark_y;
    const int probe_z = quark_z;
    
    const long probe_site = site4_index(probe_t, probe_x, probe_y, probe_z, L);

    // Compute shortest distance between quark-antiquark
    int rx = wrap_mod((antiquark_x - quark_x), L[1]);
    int ry = wrap_mod((antiquark_y - quark_y), L[2]);
    int rz = wrap_mod((antiquark_z - quark_z), L[3]);

    if(rx > (L[1]/2)) rx -=L[1];
    if(ry > (L[1]/2)) ry -=L[2];
    if(rz > (L[1]/2)) rz -=L[3];

    double magnitude_r = sqrt(rx*rx + ry*ry + rz*rz);
    if (magnitude_r == 0){
        fprintf(stderr, "ERROR: Quark and Antiquark cannot be at the same position\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double proj_weight_x = rx / magnitude_r;
    double proj_weight_y = ry / magnitude_r;
    double proj_weight_z = rz / magnitude_r;

    // Initialize E-matrices E[0]=Ex, E[1]=Ey, E[2]=Ez
    qcd_complex_16 E[3][3][3];

    for (int p = 0; p < 3; p++){
        su3_traceless_F_from_Q(clv.D[probe_site][p], E[p]);
    }

    qcd_complex_16 E_proj[3][3];
    for (int a = 0; a < 3; a++){
        for (int b = 0; b < 3; b++){
            E_proj[a][b].re = proj_weight_x * E[0][a][b].re + proj_weight_y * E[1][a][b].re + proj_weight_z * E[2][a][b].re;

            E_proj[a][b].im = proj_weight_x * E[0][a][b].im + proj_weight_y * E[1][a][b].im + proj_weight_z * E[2][a][b].im;
        }
    }

    if (myid == 0) {
        printf("Quark - Antiquark Off-axis Separation Configuration:\n");
        printf("Quark Coordinates (%d, %d, %d), Antiquark Coordinates (%d, %d, %d)\n", quark_x, quark_y, quark_z, antiquark_x, antiquark_y, antiquark_z);
        printf("Separation r: %.8e\n", magnitude_r);
        printf("Projected Chromoelectric Field (E_proj) at Quark Source:\n");
        for (int a = 0; a < 3; a++){
                for (int b = 0; b < 3; b++) {
                    printf("E_proj[%d][%d] = %.16e %.16ei\n", 
                        a, b,
                        E_proj[a][b].re, E_proj[a][b].im);
                }
            }
        fflush(stdout);
    }

    qcd_destroyCloverField(&clv);
    qcd_destroyGaugeField(&u);
    qcd_destroyGeometry(&geo);

    MPI_Finalize();
    return EXIT_SUCCESS;
}


