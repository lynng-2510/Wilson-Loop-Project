/* Computing the Laplacian correlator L(R,T) with E insertion on the temporal line at Quark source */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <math.h>
#include <qcd.h>
#include <DistEigvecsHdf5Reader.h>

#define NV 10
#define DEFAULT_RMAX 3
#define DEFAULT_TMAX 4

static int site3_index(const int x, const int y, const int z, const int L[4]){
    return (x * L[2] + y) * L[3] + z;
}

static int site4_index(const int t, const int x, const int y, const int z, const int L[4]){
    return ((t * L[1] + x) * L[2] + y) * L[3] + z;
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

// struct spatial_coordinates{
//     const int x;
//     const int y;
//     const int z;
// };

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

typedef enum {
    T0_MODE_FIXED = 0,
    T0_MODE_AVERAGE = 1,
} t0_mode_t;

static t0_mode_t read_t0_mode(void){
    const char *text = getenv("T0_MODE");
    if (text == NULL || *text == '\0' || strcmp(text, "fixed") == 0){
        return T0_MODE_FIXED;
    }

    if(strcmp(text, "average") == 0){
        return T0_MODE_AVERAGE;
    }

    fprintf(stderr, "ERROR: T0_MODE must be 'fixed' or 'average', got '%s'\n", text);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    return T0_MODE_FIXED; // return to satisfy the compiler
}

static const char *t0_mode_name(const t0_mode_t mode) {
    return (
        mode == T0_MODE_AVERAGE ? "average" : "fixed");
}

// Function to read the eigenvectors
static void read_evec_time(
    const char *fname,
    const int nc, 
    const int time,
    const char *nbase,
    qcd_geometry *geo,
    qcd_spinorComponent3d *v
) {
    DistEigvecsHdf5Reader reader;
    memset(&reader, 0, sizeof(reader));
    DistEigvecsHdf5_ComplexDouble *disteigvecs = NULL;
    int ret;
    char *tmp_run_name = NULL;
    int tmp_nc = -1;
    int tmp_time = -1;
    unsigned int tmp_nv = 0;
    int spatial_lat_size[3] = {geo->L[1], geo->L[2], geo->L[3]};
    int spatial_proclat_size[3] = {
        geo->L[1]/geo->lL[1],
        geo->L[2]/geo->lL[2],
        geo->L[3]/geo->lL[3]
    };
    int spatial_proclat_pos[3] = {geo->Pos[1], geo->Pos[2], geo->Pos[3]};

    DistEigvecsHdf5_Metadata metadata;
    DistEigvecsHdf5_Programdata programdata;
    
    ret = DistEigvecsHdf5Reader_Init(&reader);
    if(ret != 0){
        fprintf(stderr, "ERROR: DistEigvecsHdf5Reader_Init failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    ret = DistEigvecsHdf5Reader_OpenFile(
        &reader,
        geo->comm3d,
        fname,
        &tmp_run_name,
        &tmp_nc,
        spatial_lat_size,
        spatial_proclat_size,
        &tmp_time,
        &tmp_nv,
        &metadata,
        &programdata
    );
    if (ret != 0) {
        fprintf(stderr, "ERROR: OpenFile failed for %s\n", fname);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if(tmp_nc != nc || tmp_time != time || tmp_nv < NV || strcmp(tmp_run_name, nbase) !=0){
        fprintf(stderr,
                "ERROR: eigenvector metadata mismatch for %s: tmp_nc=%d tmp_time=%d tmp_nv=%u tmp_run_name=%s\n",
                fname, tmp_nc, tmp_time, tmp_nv, tmp_run_name ? tmp_run_name : "(null)");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Allocate NV eigenvector[site3][color]
    disteigvecs = malloc((size_t)geo->lV3 * 3 * NV * sizeof(DistEigvecsHdf5_ComplexDouble));
    if(!disteigvecs){
        fprintf(stderr, "ERROR: malloc disteigvecs failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    ret = DistEigvecsHdf5Reader_ReadDistEigvecsRange(
        &reader,
        spatial_proclat_pos,
        0,
        NV,
        disteigvecs
    );

    if(ret !=0) {
        fprintf(stderr, "ERROR: ReadDistEigvecsRange failed for %s\n", fname);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    size_t ncp = (size_t)geo->lV3 * 3 * sizeof(DistEigvecsHdf5_ComplexDouble);
    for (int j = 0; j < NV ; j++){
        memcpy(&(v[j].D[0][0].re), &(disteigvecs[(size_t)j * geo->lV3 * 3]), ncp);
    }
 
    free(disteigvecs);

    ret = DistEigvecsHdf5Reader_CloseFile(&reader);
    if (ret != 0) {
        fprintf(stderr, "ERROR: CloseFile failed for %s\n", fname);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    DistEigvecsHdf5Reader_Free(&reader);
}

/* matrix-vector multiplication function, computing 
output[a] = sum of matrix[a][b] * input[b] */
static void su3_mat_vec(
    const qcd_complex_16 matrix[3][3], // gauge field or E (3x3)
    const qcd_complex_16 input[3],
    qcd_complex_16 output[3]
) {
    for (int a = 0; a < 3; a++) {
        output[a].re = 0.0;
        output[a].im = 0.0;

        for (int b = 0; b < 3; b++) {
            output[a] = qcd_CADD(
                output[a],
                qcd_CMUL(matrix[a][b], input[b])
            );
        }
    }
}

/* output_dagger = conjugate of output */
static void su3_dagger_mat_vec(
    const qcd_complex_16 matrix[3][3],
    const qcd_complex_16 input[3],
    qcd_complex_16 output[3]
) {
    for (int a = 0; a < 3; a++) {
        output[a].re = 0.0;
        output[a].im = 0.0;

        for (int b = 0; b < 3; b++) {
            output[a] = qcd_CADD(
                output[a],
                qcd_CMUL(qcd_CONJ(matrix[b][a]), input[b])
            );
        }
    }
}

// Function to compute the perambulator for anti-quark without E insertion
static qcd_complex_16 tau_time_path(
    const qcd_spinorComponent3d *v_src,
    const qcd_spinorComponent3d *v_sink,
    const qcd_gaugeField *u,
    const int i,
    const int j,
    const int t0,
    const int T,
    const int site3,
    const int x,
    const int y,
    const int z,
    const int L[4]
){
    qcd_complex_16 vec[3];
    qcd_complex_16 tmp[3];

    for (int a = 0; a < 3; a++){
        vec[a] = v_sink[j].D[site3][a];
    }

    /*
    Compute the forward temporal transporter

        U_0(x,t0) U_0(x,t0+1) ... U_0(x,t0+T-1)
        v_j(x,t0+T),

      with periodic temporal wrapping.

      The matrices are applied backwards to the sink vector.
    */
    for (int step = T - 1; step >= 0; step--) {
    const int tt = (t0 + step) % L[0];
    const int site4 = site4_index(
        tt,
        x,
        y,
        z,
        L
    );

        su3_mat_vec(u->D[site4][0], vec, tmp);
        memcpy(vec, tmp, sizeof vec);
    }

    qcd_complex_16 tau = {0.0, 0.0};

    for (int a = 0; a < 3; a++) {
        tau = qcd_CADD(
            tau,
            qcd_CMUL(
                qcd_CONJ(v_src[i].D[site3][a]),
                vec[a]
            )
        );
    }  
    return tau;
}

// Function to compute the perambulator with E insertion on the temporal line at Quark source
static qcd_complex_16 tau_time_path_E_insertion(
    const qcd_spinorComponent3d *v_src,
    const qcd_spinorComponent3d *v_sink,
    const qcd_gaugeField *u,
    const int i,
    const int j,
    const int t0,
    const int T, // T odd and even
    const int insert_step,
    const int site3,
    const int x,
    const int y,
    const int z,
    const int L[4],
    qcd_complex_16 E_insert[3][3]
){
    qcd_complex_16 vec[3];
    qcd_complex_16 tmp[3];

    if (T <= 0 || insert_step < 0 || insert_step > T){
        fprintf(stderr, "ERROR: invalid T or insertion step\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int a = 0; a < 3; a++){
        vec[a] = v_src[i].D[site3][a];
    }

    /*
    First half of the reverse temporal path:

    vec =
        U0†(x, t0 + T/2 - 1) ...
        U0†(x, t0 + 1)
        U0†(x, t0)
        vi(x, t0).
    
    The matrices are applied forward from the source vector.
    */
    for (int step = 0; step < insert_step; step++){
        const int tt = (t0 + step) % L[0];
        const int site4 = site4_index(tt, x, y, z, L);
        su3_dagger_mat_vec(u->D[site4][0], vec, tmp);
        memcpy(vec, tmp, sizeof vec);
    }

    /* Insert E matrix at t0 + insert step */
    su3_mat_vec(E_insert, vec, tmp);
    memcpy(vec, tmp, sizeof vec);

    /* 
    Second half of the reverse temporal path:

    U†(t0 + T - 1) ... U†(t0 + insert_step)
    */
    for (int step = insert_step; step < T; step++){
        const int tt = (t0 + step) % L[0];
        const int site4 = site4_index(tt, x, y, z, L);
        su3_dagger_mat_vec(u->D[site4][0], vec, tmp);
        memcpy(vec, tmp, sizeof vec);    
    }

    qcd_complex_16 tau_E = {0.0, 0.0};
    for (int a = 0; a < 3; a++){
        tau_E = qcd_CADD(
            tau_E, 
            qcd_CMUL(qcd_CONJ(v_sink[j].D[site3][a]), vec[a]));
    }
    return tau_E;
}     

// Function handling insert steps for case T odd and even
static int get_insert_steps(const int T, int insert_steps[2]) {
    if (T % 2 == 0) {
        insert_steps[0] = T / 2;
        return 1;
    }
    insert_steps[0] = T / 2;
    insert_steps[1] = T / 2 + 1;
    return 2;
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int myid = 0;
    int numprocs = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

if (numprocs != 1) {
        if (myid == 0) {
            fprintf(
                stderr,
                "ERROR: this test is intended for exactly 1 MPI rank.\n"
            );
        }

        MPI_Abort(
            MPI_COMM_WORLD, EXIT_FAILURE
        );
    }

    const char *nbase = "Em1p4";

    int L[4] = {48, 24, 24, 24};
    int P[4] = {1, 1, 1, 1};

    const int nc = read_env_int(
        "NCFG",
        1,
        1,
        INT_MAX
    );

    const int rmax = read_env_int(
        "RMAX",
        DEFAULT_RMAX,
        0,
        L[1] / 2
    );

    const int tmax = read_env_int(
        "TMAX",
        DEFAULT_TMAX,
        1,
        L[0] - 1
    );

    const t0_mode_t t0_mode = read_t0_mode();

    const int t0_fixed = read_env_int(
        "T0_FIXED",
        0,
        0,
        L[0] - 1
    );

    const int t0_start = read_env_int(
        "T0_START",
        0,
        0,
        L[0] - 1
    );

    const int t0_stride = read_env_int(
        "T0_STRIDE",
        1,
        1,
        L[0]
    );

    int t0_values[L[0]];
    int nt0 = 0;

    if (t0_mode == T0_MODE_FIXED) {
        t0_values[nt0++] = t0_fixed;
    } else {
        for (int t0 = t0_start;
        t0 < L[0];
        t0 += t0_stride
    ) { 
        t0_values[nt0++] = t0;
    }
}
if (nt0 < 1) {
    fprintf(stderr, "ERROR: no temporal source times selected\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

unsigned char needed_time[L[0]];
unsigned char initialized_time[L[0]];

memset(needed_time, 0, sizeof(needed_time));
memset(initialized_time, 0, sizeof(initialized_time));

for (int source = 0; source < nt0; source++) {
        const int t0 = t0_values[source];

        needed_time[t0] = 1;

        for (int T = 1; T <= tmax; T++) {
            const int t1 = (t0 + T) % L[0];
            needed_time[t1] = 1;
        }
    }

    int n_needed_times = 0;

    for (int t = 0; t < L[0]; t++) {
        if (needed_time[t]) {
            n_needed_times++;
        }
    }

    double theta[3] = {0.0, 0.0, 0.0};
    qcd_geometry geo;
    qcd_gaugeField u;
    qcd_cloverField clv;

    // Geometry initialization
    if (qcd_initGeometry(
            &geo,
            L,
            P,
            theta,
            myid,
            numprocs
        )
    ) {
        fprintf(stderr, "ERROR: qcd_initGeometry failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Even-odd geometry
    if (qcd_initEO(&geo)) {
        fprintf(stderr, "ERROR: qcd_initEO failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Gauge field initialization
    if (qcd_initGaugeField(&u, &geo)) {
        fprintf(stderr,"ERROR: qcd_initGaugeField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Clover field initialization
    if (qcd_initCloverField(&clv, &geo)) {
        fprintf(stderr, "ERROR: qcd_initCloverField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    qcd_spinorComponent3d *v_by_time = calloc((size_t)L[0] * NV, sizeof(*v_by_time));

    if (v_by_time == NULL){
        fprintf(stderr, "ERROR: allocation of temporal eigenvectors failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char gauge_field_path[1024];
    snprintf(
        gauge_field_path, 
        sizeof(gauge_field_path),
        "/home/m2427382/Wilson-Loop-Project/Em1/cnfg/Em1p4n%d", nc);
    
    char evec_file[1024];
    if (myid == 0) {
        printf(
            "Reading gauge field: %s\n",
            gauge_field_path
        );
        fflush(stdout);
    }
    // Gauge file loading
    if (
        qcd_getGaugeField(gauge_field_path, qcd_GF_OPENQCD, &u )
    ) {
        fprintf(stderr, "ERROR: qcd_getGaugeField failed\n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const double plaq = qcd_calculatePlaquette(&u);

    if (myid == 0){
        printf("Average plaquette = %.16e\n", plaq);
        fflush(stdout);
    }
    // Clover tensor calculation
    if (qcd_calculateCloverField(&clv, &u)){
        fprintf(stderr, "ERROR: qcd_calculateCloverField failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int t = 0; t < L[0]; t++) {
        if (!needed_time[t]) { // Skip loading HDF5 files for time different from t
            continue;
        }
        qcd_spinorComponent3d *v_time = (
            &v_by_time[(size_t)t * NV]
        );
        for (int i = 0; i < NV; i++){
            if (
                qcd_initSpinorComponent3d(
                    &v_time[i],
                    &geo
                )
            ){
                fprintf(stderr,"ERROR: qcd_initSpinorComponent3d failed at t=%d i=%d\n",
                        t, i);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        initialized_time[t] = 1;
        
        snprintf(
            evec_file,
            sizeof(evec_file),
            "/home/m2427382/Wilson-Loop-Project/Em1/cnfg/n%d/eigenvectors/Em1p4n%d_evec_t%d.h5",
            nc,
            nc,
            t
        );

        if (myid == 0) {
            printf("Reading temporal eigenvectors t=%d: %s\n", t, evec_file);
            fflush(stdout);
        }

        read_evec_time(evec_file, nc, t, nbase, &geo, v_time);
    }
    if (myid == 0){
        printf(
            "\n RT source-average scan: "
            "cfg=%d Nv=%d R/a=0..%d T/a=1..%d "
            "T0Mode=%s Nt0=%d rho_i=1\n",
            nc, NV, rmax, tmax, t0_mode_name(t0_mode), nt0);

        printf("META cfg=%d Lattice dimension %d x %d x %d x %d\n",
            nc, L[0], L[1], L[2], L[3]
        );

        printf("Nv=%d Rmin=0 Rmax=%d "
            "Tmin=1 Tmax=%d T0Mode=%s Nt0=%d "
            "T0Fixed=%d T0Start=%d T0Stride=%d "
            "NeededTimes=%d temporal_bc=periodic "
            "rho=constant\n",
            NV, rmax, tmax, t0_mode_name(t0_mode), nt0,
            t0_fixed, t0_start, t0_stride, n_needed_times
        );
        
        printf("T0_LIST");
        for (int source = 0; source < nt0; source++) {
            printf("[%d]", t0_values[source]);
        }

        printf("\n");
        fflush(stdout);
    }

    for (int T = 4; T <= tmax; T++){
        for (int R = 3; R <= rmax; R++) {
            
            int rx = 0 + R;
            int ry = 0;
            int rz = 0;
            double magnitude_r = sqrt(rx*rx + ry*ry + rz*rz);

            if (magnitude_r == 0){
                fprintf(stderr, "ERROR: Quark and Antiquark cannot be at the same position\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            int probe_t[2];
            int num_insertions = get_insert_steps(T, probe_t);

            if (myid == 0) {
                printf("[RUN] T=%d, R=%d Vector_r=(%d,%d,%d) |r|=%.3f\n", 
                        T, R, rx, ry, rz, magnitude_r);
                if (num_insertions == 1) {
                    printf("E-insertion at T-insert=%d\n", T / 2);
                } else {
                    printf("E-insertion at T-insert=%d, t=%d\n", T / 2, T / 2 + 1);
                }
                fflush(stdout);
            }

            qcd_complex_16 Lsum = {0.0, 0.0};
            long nsrc = 0;
            for (int source = 0; source < nt0; source++){
                const int t0 = t0_values[source];
                const int t1 = (t0 + T) % L[0];

                const qcd_spinorComponent3d *v_src = (&v_by_time[(size_t)t0 * NV]);
                const qcd_spinorComponent3d *v_sink = (&v_by_time[(size_t)t1 * NV]);

                // Spatial loop sweeps across L[1]xL[2]xL[3] lattice origin sites
                for (int xx = 0; xx < L[1]; xx++) {
                    for (int yy = 0; yy < L[2]; yy++){
                        for (int zz = 0; zz < L[3]; zz++){
                            const int shifted_x = (xx + rx) % L[1];
                            const int shifted_y = (yy + ry) % L[2];
                            const int shifted_z = (zz + rz) % L[3];
                            
                            const int x_site3 = site3_index(xx, yy, zz, L);
                            const int y_site3 = site3_index(shifted_x, shifted_y, shifted_z, L);
                            
                            double proj_weight_x = rx / magnitude_r;
                            double proj_weight_y = ry / magnitude_r;
                            double proj_weight_z = rz / magnitude_r;

                            const int probe_x = xx;
                            const int probe_y = yy;
                            const int probe_z = zz;

                            long probe_site[2];
                            for (int i = 0; i < num_insertions; i ++){
                                int real_probe_t = (t0 + probe_t[i]) % L[0];
                                probe_site[i] = site4_index(real_probe_t, probe_x, probe_y, probe_z, L);
                            }
                            qcd_complex_16 Lxy_part[2];
                            Lxy_part[0].re = 0.0; Lxy_part[0].im = 0.0;
                            Lxy_part[1].re = 0.0; Lxy_part[1].im = 0.0;

                            for (int step_idx = 0; step_idx < num_insertions; step_idx++){
                            
                                // Initialize E-matrices E[0]=Ex, E[1]=Ey, E[2]=Ez
                                qcd_complex_16 E[3][3][3];
                                for (int p = 0; p < 3; p++){
                                        su3_traceless_F_from_Q(clv.D[probe_site[step_idx]][p], E[p]);
                                    }

                                // Compute the inserted E_projection    
                                qcd_complex_16 E_proj[3][3];
                                for (int a = 0; a < 3; a++){
                                    for (int b = 0; b < 3; b++){
                                        E_proj[a][b].re = proj_weight_x * E[0][a][b].re + proj_weight_y * E[1][a][b].re + proj_weight_z * E[2][a][b].re;

                                        E_proj[a][b].im = proj_weight_x * E[0][a][b].im + proj_weight_y * E[1][a][b].im + proj_weight_z * E[2][a][b].im;
                                    }
                                }

                                // // Test when E = I
                                // qcd_complex_16 E_proj[3][3];
                                // for (int a = 0; a < 3; a++){
                                //     for (int b = 0; b < 3; b++){
                                //         E_proj[a][b].re = (a == b) ? 1.0 : 0.0; // set value 1 for diagonal, 0 for off-diagonal
                                //         E_proj[a][b].im = 0.0;
                                //     }
                                // }

                                // Computing Laplacian correlators L
                                for (int i = 0; i < NV; i++){
                                    for (int j = 0; j < NV; j++){

                                        const qcd_complex_16 tau_y = tau_time_path(v_src, v_sink, &u, i, j, t0, T, y_site3, shifted_x, shifted_y, shifted_z, L);
                            
                                        const qcd_complex_16 tau_E_x = tau_time_path_E_insertion(v_src, v_sink, &u, i, j, t0, T, probe_t[step_idx], x_site3, xx, yy, zz, L, E_proj);

                                        Lxy_part[step_idx] = qcd_CADD(Lxy_part[step_idx], qcd_CMUL(tau_y, tau_E_x));
                                        }
                                    }
                            }
                            qcd_complex_16 final_Lxy = {0.0, 0.0};
                            // When T is even
                            if (num_insertions == 1){
                                final_Lxy = Lxy_part[0];
                            } 
                            // When T is odd
                            else if (num_insertions == 2) {
                                final_Lxy.re = (Lxy_part[0].re + Lxy_part[1].re) * 0.5;
                                final_Lxy.im = (Lxy_part[0].im + Lxy_part[1].im) * 0.5;
                            }

                            Lsum = qcd_CADD(Lsum, final_Lxy);
                            nsrc++;
                        }
                    }
                }
            }
            // Compute average correlator for a pair (T,R)
            const qcd_complex_16 Lavg = {
                Lsum.re / (double)nsrc, 
                Lsum.im / (double)nsrc};

            if (myid == 0) {
                printf(
                    "# DATA cfg T R Nsrc "
                    "Re[avg L(R,T)] Im[avg L(R,T)]\n"
                );
                printf(
                    "DATA %d %d %d %ld %+.16e %+.16e\n",
                    nc, T, R, nsrc,
                    Lavg.re, Lavg.im
                );
                fflush(stdout);
            }
        }
    }

    for (int t = 0; t < L[0]; t++) {
        if (!initialized_time[t]) {
            continue;
        }

        qcd_spinorComponent3d *v_time = (
            &v_by_time[(size_t)t * NV]
        );

        for (int i = 0; i < NV; i++) {
            qcd_destroySpinorComponent3d(
                &v_time[i]
            );
        }
    }

    free(v_by_time);

    qcd_destroyCloverField(&clv);
    qcd_destroyGaugeField(&u);
    qcd_destroyEO(&geo);
    qcd_destroyGeometry(&geo);

    MPI_Finalize();

    return 0;
}
