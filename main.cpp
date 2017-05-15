#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

using namespace std;


// Neodnorodnost'
double F(double x, double y, double z) {
    return 3 * exp(x + y + z);
}

// Granichnoe uslovie pri x=0
double A0(double y, double z) {
    return exp(y + z);
}

// Granichnoe uslovie pri x=X
double A1(double y, double z, double X) {
    return exp(X + y + z);
}

// Granichnoe uslovie pri y=0
double B0(double x, double z) {
    return exp(x + z);
}

// Granichnoe uslovie pri y=Y
double B1(double x, double z, double Y) {
    return exp(x + Y + z);
}

// Granichnoe uslovie pri z=0
double C0(double x, double y) {
    return exp(x + y);
}

// Granichnoe uslovie pri z=Z
double C1(double x, double y, double Z) {
    return exp(x + y + Z);
}

bool calculationsNeeded(int igl1, int igl2, int igl3, int igl4,
                        int rit, int Nx, int Ny, int Nz,
                        int r1, int r2, int r3, int r4) {
    for (int i1 = igl1 * r1 + 1; i1 < min((igl1 + 1) * r1 + 1, rit + 1); ++i1) {
        if (max(0, i1 - igl2 * r2) < min(r2, i1 + Nx - igl2 * r2)
            && max(0, i1 - igl3 * r3) < min(r3, i1 + Ny - igl3 * r3)
            && max(igl4 * r4, i1) < min((igl4 + 1) * r4, i1 + Nz)) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {
    int rank;
    int dim[2], period[2], reorder, coord[2];
    dim[0] = atoi(argv[1]); dim[1] = atoi(argv[2]);
    int up, down, right, left, upLeft, downRight;
    double start, end;

    MPI_Comm grid_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    period[0] = false; period[1] = false;
    reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);

    // Nahozhedie sosednih processov
    MPI_Cart_shift(grid_comm, 0, 1, &left, &right);
    MPI_Cart_shift(grid_comm, 1, 1, &up, &down);

    int neighbour[2][2];
    if (up != -1 && left != -1) {
        MPI_Cart_coords(grid_comm, up, 2, neighbour[0]);
        MPI_Cart_coords(grid_comm, left, 2, neighbour[1]);
        neighbour[0][0] = neighbour[1][0];
        MPI_Cart_rank(grid_comm, neighbour[0], &upLeft);
    } else {
        upLeft = -1;
    }

    if (down != -1 && right != -1) {
        MPI_Cart_coords(grid_comm, down, 2, neighbour[0]);
        MPI_Cart_coords(grid_comm, right, 2, neighbour[1]);
        neighbour[0][0] = neighbour[1][0];
        MPI_Cart_rank(grid_comm, neighbour[0], &downRight);
    } else {
        downRight = -1;
    }

    // Vichislenie osnovnih parametrov oblasti reshenija
    int rit = 200, tag = 1000;
    double h1 = 0.01, h2 = 0.01, h3 = 0.01;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r1 = 10, r4 = 20;
    int Q1 = (int)ceil((double)rit / r1);
    int Q4 = (int)ceil((double)(rit + Nz) / r4);

    int Q2 = dim[0], Q3 = dim[1];
    int r2 = (int)ceil((double)(rit + Nx) / Q2);
    int r3 = (int)ceil((double)(rit + Ny) / Q3);

    double *U = (double *)calloc((size_t)(r1 + 1) * r2 * r3 * r4 * Q4, sizeof(double));
    double *preBack = (double *)calloc((size_t)(r1 + 1) * r2 * r4 * Q4, sizeof(double));
    double *preLeft = (double *)calloc((size_t)(r1 + 1) * r3 * r4 * Q4, sizeof(double));
    double *preBackLeft = (double *)calloc((size_t) (r1 + 1) * r4 * Q4, sizeof(double));

    MPI_Cart_coords(grid_comm, rank, 2, coord);
    int igl2 = coord[0];
    int igl3 = coord[1];

    int igl2_l, igl2_u, igl2_ul;
    int igl3_l, igl3_u, igl3_ul;
    if (left != -1) {
        MPI_Cart_coords(grid_comm, left, 2, coord);
        igl2_l = coord[0];
        igl3_l = coord[1];
    }
    if (up != -1) {
        MPI_Cart_coords(grid_comm, up, 2, coord);
        igl2_u = coord[0];
        igl3_u = coord[1];
    }
    if (upLeft != -1) {
        MPI_Cart_coords(grid_comm, upLeft, 2, coord);
        igl2_ul = coord[0];
        igl3_ul = coord[1];
    }

    MPI_Status status;

    MPI_Datatype uik_t;
    MPI_Type_vector(r2, r4, r3 * r4 * Q4, MPI_DOUBLE, &uik_t);
    MPI_Type_commit(&uik_t);
    MPI_Type_create_resized(uik_t, 0, sizeof(double) * r2 * r3 * r4 * Q4, &uik_t);
    MPI_Type_commit(&uik_t);

    MPI_Datatype ujk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);
    MPI_Type_create_resized(ujk_t, 0, sizeof(double) * r2 * r3 * r4 * Q4, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype uk_t;
    MPI_Type_vector(r1, r4, r2 * r3 * r4 * Q4, MPI_DOUBLE, &uk_t);
    MPI_Type_commit(&uk_t);

    MPI_Datatype preik_t;
    MPI_Type_vector(r2, r4, r4 * Q4, MPI_DOUBLE, &preik_t);
    MPI_Type_commit(&preik_t);
    MPI_Type_create_resized(preik_t, 0, sizeof(double) * r2 * r4 * Q4, &preik_t);
    MPI_Type_commit(&preik_t);

    MPI_Datatype  prejk_t;
    MPI_Type_vector(r3, r4, r4 * Q4, MPI_DOUBLE, &prejk_t);
    MPI_Type_commit(&prejk_t);
    MPI_Type_create_resized(prejk_t, 0, sizeof(double) * r3 * r4 * Q4, &prejk_t);
    MPI_Type_commit(&prejk_t);

    MPI_Datatype  prek_t;
    MPI_Type_vector(r1, r4, r4 * Q4, MPI_DOUBLE, &prek_t);
    MPI_Type_commit(&prek_t);

    for (int igl1 = 0; igl1 < Q1; ++igl1) {
        for (int igl4 = 0; igl4 < Q4; ++igl4) {
            if (upLeft != -1) {
                bool upLeftRecvNeeded = calculationsNeeded(igl1, igl2_ul, igl3_ul, igl4,
                                                           rit, Nx, Ny, Nz, r1, r2, r3, r4);
                if (upLeftRecvNeeded) {
//                    printf("%d (%d, %d) receives upLeft from %d\n", rank, igl1, igl4, upLeft);
                    MPI_Recv(preBackLeft + (Q4 + igl4) * r4, 1, prek_t, upLeft, tag * igl1 + igl4, grid_comm, &status);
//                    printf("%d (%d, %d) received upLeft from %d\n", rank, igl1, igl4, upLeft);
                }
            }

            if (left != -1) {
                bool leftRecvNeeded = calculationsNeeded(igl1, igl2_l, igl3_l, igl4, rit, Nx, Ny, Nz, r1, r2, r3, r4);
                if (leftRecvNeeded) {
//                    printf("%d (%d, %d) receives left from %d\n", rank, igl1, igl4, left);
                    MPI_Recv(preLeft + (r3 * Q4 + igl4) * r4, r1, prejk_t, left, tag * igl1 + igl4, grid_comm,
                             &status);
//                    printf("%d (%d, %d) received left from %d\n", rank, igl1, igl4, left);
                }
            }


            if (up != -1) {
                bool upRecvNeeded = calculationsNeeded(igl1, igl2_u, igl3_u, igl4, rit, Nx, Ny, Nz, r1, r2, r3, r4);
                if (upRecvNeeded) {
//                    printf("%d (%d, %d) receives up from %d\n", rank, igl1, igl4, up);
                    MPI_Recv(preBack + (r2 * Q4 + igl4) * r4, r1, preik_t, up, tag * igl1 + igl4, grid_comm, &status);
//                    printf("%d (%d, %d) received up from %d\n", rank, igl1, igl4, up);
                }
            }

            bool sendNeeded = false;
//            printf("%d (%d, %d) enters tile\n", rank, igl1, igl4);
            for (int i1 = igl1 * r1 + 1, ii1 = 1; i1 < min((igl1 + 1) * r1 + 1, rit + 1); ++i1, ++ii1) {
                for (int i2 = max(0, i1 - igl2 * r2); i2 < min(r2, i1 + Nx - igl2 * r2); ++i2) {
                    for (int i3 = max(0, i1 - igl3 * r3); i3 < min(r3, i1 + Ny - igl3 * r3); ++i3) {
                        for (int i4 = max(igl4 * r4, i1); i4 < min((igl4 + 1) * r4, i1 + Nz); ++i4) {
                            sendNeeded = true;

                            double uip, uim, ujp, ujm, ukp, ukm, u;
                            int i = igl2 * r2 + i2 - i1, j = igl3 * r3 + i3 - i1, k = i4 - i1;

                            if (i == 0) {
                                uim = A0((j + 1) * h2, (k + 1) * h3);
                            } else if (i2 == 0) {
                                uim = preLeft[(ii1 * r3 + i3) * r4 * Q4 + i4];
                            } else {
                                uim = U[((ii1 * r2 + i2 - 1) * r3 + i3) * r4 * Q4 + i4];
                            }

                            if (i == Nx - 1) {
                                uip = A1((j + 1) * h2, (k + 1) * h3, X);
                            } else if (i3 == 0) {
                                uip = preBack[((ii1 - 1) * r2 + i2) * r4 * Q4 + i4 - 1];
                            } else {
                                uip = U[(((ii1 - 1) * r2 + i2) * r3 + i3 - 1) * r4 * Q4 + i4 - 1];
                            }


                            if (j == 0) {
                                ujm = B0((i + 1) * h1, (k + 1) * h3);
                            } else if (i3 == 0) {
                                ujm = preBack[(ii1 * r2 + i2) * r4 * Q4 + i4];
                            } else {
                                ujm = U[((ii1 * r2 + i2) * r3 + i3 - 1) * r4 * Q4 + i4];
                            }

                            if (j == Ny - 1) {
                                ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                            } else if (i2 == 0) {
                                ujp = preLeft[((ii1 - 1) * r3 + i3) * r4 * Q4 + i4 - 1];
                            } else {
                                ujp = U[(((ii1 - 1) * r2 + i2 - 1) * r3 + i3) * r4 * Q4 + i4 - 1];
                            }

                            if (k == 0) {
                                ukm = C0((i + 1) * h1, (j + 1) * h2);
                            } else {
                                ukm = U[((ii1 * r2 + i2) * r3 + i3) * r4 * Q4 + i4 - 1];
                            }

                            if (k == Nz - 1) {
                                ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                            } else if (i2 == 0) {
                                if (i3 == 0) {
                                    ukp = preBackLeft[(ii1 - 1) * r4 * Q4 + i4];
                                } else {
                                    ukp = preLeft[((ii1 - 1) * r3 + i3 - 1) * r4 * Q4 + i4];
                                }
                            } else {
                                if (i3 == 0) {
                                    ukp = preBack[((ii1 - 1) * r2 + i2  - 1) * r4 * Q4 + i4];
                                } else {
                                    ukp = U[(((ii1 - 1) * r2 + i2 - 1) * r3 + i3 - 1) * r4 * Q4 + i4];
                                }
                            }

                            if (i2 == 0) {
                                if (i3 == 0) {
                                    u = preBackLeft[(ii1 - 1) * r4 * Q4 + i4 - 1];
                                } else {
                                    u = preLeft[((ii1 - 1) * r3 + i3 - 1) * r4 * Q4 + i4 - 1];
                                }
                            } else {
                                if (i3 == 0) {
                                    u = preBack[((ii1 - 1) * r2 + i2  - 1) * r4 * Q4 + i4 - 1];
                                } else {
                                    u = U[(((ii1 - 1) * r2 + i2 - 1) * r3 + i3 - 1) * r4 * Q4 + i4 - 1];
                                }
                            }

                            U[((ii1 * r2 + i2) * r3 + i3) * r4 * Q4 + i4] =
                                    w * ((uip + uim) / (h1 * h1)
                                         + (ujp + ujm) / (h2 * h2)
                                         + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                       (j + 1) * h2,
                                                                       (k + 1) * h3)) /
                                    (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3)) + (1 - w) * u;
                        }
                    }
                }
            }
//            printf("%d (%d, %d) leaves tile\n", rank, igl1, igl4);

            if (sendNeeded) {
                if (downRight != -1) {
//                    printf("%d (%d, %d) Isends downRight to %d \n", rank, igl1, igl4, downRight);
                    MPI_Send(U + ((2 * r2 * r3 - 1) * Q4 + igl4) * r4, 1, uk_t,
                              downRight, tag * igl1 + igl4, grid_comm);
//                    printf("%d (%d, %d) Isent downRight to %d \n", rank, igl1, igl4, downRight);
                }

                if (down != -1) {
//                    printf("%d (%d, %d) Isends down to %d \n", rank, igl1, igl4, down);
                    MPI_Send(U + (((r2 + 1) * r3 - 1) * Q4 + igl4) * r4, r1, uik_t,
                              down, tag * igl1 + igl4, grid_comm);
//                    printf("%d (%d, %d) Isent down to %d \n", rank, igl1, igl4, down);
                }

                if (right != - 1) {
//                    printf("%d (%d, %d) Isends right to %d \n", rank, igl1, igl4, right);
                    MPI_Send(U + ((2 * r2 - 1) * r3 * Q4 + igl4) * r4, r1, ujk_t,
                              right, tag * igl1 + igl4, grid_comm);
//                    printf("%d (%d, %d) Isent right to %d \n", rank, igl1, igl4, right);
                }
            }
        }

        if (igl1 != Q1 - 1) {
            memcpy(U, U + r1 * r2 * r3 * r4 * Q4, r2 * r3 * r4 * Q4 * sizeof(double));
            memcpy(preLeft, preLeft + r1 * r3 * r4 * Q4, r3 * r4 * Q4 * sizeof(double));
            memcpy(preBack, preBack + r1 * r2 * r4 * Q4, r2 * r4 * Q4 * sizeof(double));
            memcpy(preBackLeft, preBackLeft + r1 * r4 * Q4, r4 * Q4 * sizeof(double));
        }
    }

    double *R;
    int l = rit % r1;
    if (l == 0) {
        l = r1;
    }

    if (rank == 0) {
        R = (double *)malloc(sizeof(double) * r2 * Q2 * r3 * Q3 * r4 * Q4);

        MPI_Datatype r_t;
        MPI_Type_vector(r2, r3 * r4 * Q4, r3 * Q3 * r4 * Q4, MPI_DOUBLE, &r_t);
        MPI_Type_commit(&r_t);

        int self[2];
        MPI_Cart_coords(grid_comm, rank, 2, self);
        for (int n = 0; n < dim[0]; ++n) {
            for (int m = 0; m < dim[1]; ++m) {
                if (n == self[0] && m == self[1]) {
                    for (int i = 0; i < r2; ++i) {
                        for (int j = 0; j < r3; ++j) {
                            for (int k = 0; k < r4 * Q4; ++k) {
                                R[(((r2 * n + i) * Q3 + r3) * m + j) * r4 * Q4 + k]
                                        = U[((l * r2 + i) * r3 + j) * r4 * Q4 + k];
                            }
                        }
                    }
                    continue;
                }
                int sender;
                int coords[2];
                coords[0] = n; coords[1] = m;
                MPI_Cart_rank(grid_comm, coords, &sender);
                MPI_Recv(R + (n * r2 * Q3 + m) * r3 * r4 * Q4, 1, r_t, sender, 10005000, grid_comm, &status);
            }
        }
    } else {
        MPI_Send(U + l * r2 * r3 * r4 * Q4, r2 * r3 * r4 * Q4, MPI_DOUBLE, 0, 10005000, grid_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) {
        printf("Time: %f\n", end - start);

        FILE *f = fopen("output.txt", "w");
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    fprintf(f, "%f ", R[((rit + i) * r3 * Q3 + rit + j) * r4 * Q4 + rit + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}