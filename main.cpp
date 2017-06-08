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

bool calculationsNeeded(int m, int igl1, int igl2, int Nx, int Ny,
                        int r1, int r2) {
    return (max(0, m - igl1 * r1) < min(r1, m + Nx - igl1 * r1)
            && max(0, m - igl2 * r2) < min(r2, m + Ny - igl2 * r2));
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
    int rit = 100, tag = 31;
    double h1 = 0.01, h2 = 0.01, h3 = 0.01;
    double X = 1, Y = 1, Z = 1;
    double w = 1.7;

    int Nx = (int)(X / h1) - 1, Ny = (int)(Y / h2) - 1, Nz = (int)(Z / h3) - 1;

    int r3 = 20;
    int Q3 = (int)ceil((double)Nz / r3);

    int Q1 = dim[0], Q2 = dim[1];
    int r1 = (int)ceil((double)(rit + Nx) / Q1);
    int r2 = (int)ceil((double)(rit + Ny) / Q2);

    double *U = (double *)calloc((size_t)2 * r1 * r2 * r3 * Q3, sizeof(double));
    double *preBack = (double *)calloc((size_t)2 * r1 * r3 * Q3, sizeof(double));
    double *preLeft = (double *)calloc((size_t)2 * r2 * r3 * Q3, sizeof(double));
    double *preBackLeft = (double *)calloc((size_t)2 * r3 * Q3, sizeof(double));

    MPI_Cart_coords(grid_comm, rank, 2, coord);
    int igl1 = coord[0];
    int igl2 = coord[1];

    int igl1_l, igl1_u, igl1_ul;
    int igl2_l, igl2_u, igl2_ul;
    if (left != -1) {
        MPI_Cart_coords(grid_comm, left, 2, coord);
        igl1_l = coord[0];
        igl2_l = coord[1];
    }
    if (up != -1) {
        MPI_Cart_coords(grid_comm, up, 2, coord);
        igl1_u = coord[0];
        igl2_u = coord[1];
    }
    if (upLeft != -1) {
        MPI_Cart_coords(grid_comm, upLeft, 2, coord);
        igl1_ul = coord[0];
        igl2_ul = coord[1];
    }

    MPI_Status status;

    MPI_Datatype uik_t;
    MPI_Type_vector(r1, r3, r2 * r3 * Q3, MPI_DOUBLE, &uik_t);
    MPI_Type_commit(&uik_t);

    MPI_Datatype ujk_t;
    MPI_Type_vector(r2, r3, r3 * Q3, MPI_DOUBLE, &ujk_t);
    MPI_Type_commit(&ujk_t);

    MPI_Datatype preik_t;
    MPI_Type_vector(r1, r3, r3 * Q3, MPI_DOUBLE, &preik_t);
    MPI_Type_commit(&preik_t);

    for (int m = 1; m <= rit; ++m) {
        for (int igl3 = 0; igl3 < Q3; ++igl3) {
            if (upLeft != -1) {
                bool upLeftRecvNeeded = calculationsNeeded(m, igl1_ul, igl2_ul, Nx, Ny, r1, r2);
                if (upLeftRecvNeeded) {
//                    printf("%d (%d, %d) receives upLeft from %d\n", rank, m, igl3, upLeft);
                    MPI_Recv(preBackLeft + (Q3 + igl3) * r3, r3, MPI_DOUBLE, upLeft, m * tag + igl3,
                             grid_comm, &status);
//                    printf("%d (%d, %d) received upLeft from %d\n", rank, m, igl3, upLeft);
                }
            }

            if (left != -1) {
                bool leftRecvNeeded = calculationsNeeded(m, igl1_l, igl2_l, Nx, Ny, r1, r2);
                if (leftRecvNeeded) {
//                    printf("%d (%d, %d) receives left from %d\n", rank, m, igl3, left);
                    MPI_Recv(preLeft + (r2 * Q3 + igl3) * r3, 1, ujk_t, left, tag * m + igl3, grid_comm,
                             &status);
//                    printf("%d (%d, %d) received left from %d\n", rank, m, igl3, left);
                }
            }


            if (up != -1) {
                bool upRecvNeeded = calculationsNeeded(m, igl1_u, igl2_u, Nx, Ny, r1, r2);
                if (upRecvNeeded) {
//                    printf("%d (%d, %d) receives up from %d\n", rank, m, igl3, up);
                    MPI_Recv(preBack + (r1 * Q3 + igl3) * r3, 1, preik_t, up, tag * m + igl3, grid_comm, &status);
//                    printf("%d (%d, %d) received up from %d\n", rank, m, igl3, up);
                }
            }

            bool sendNeeded = false;
//            printf("%d (%d, %d) enters tile\n", rank, m, igl3);
            for (int i1 = max(0, m - igl1 * r1); i1 < min(r1, m + Nx - igl1 * r1); ++i1) {
                for (int i2 = max(0, m - igl2 * r2); i2 < min(r2, m + Ny - igl2 * r2); ++i2) {
                    for (int i3 = igl3 * r3; i3 < min((igl3 + 1) * r3, Nz); ++i3) {
                        sendNeeded = true;

                        double uip, uim, ujp, ujm, ukp, ukm, u;
                        int i = igl1 * r1 + i1 - m, j = igl2 * r2 + i2 - m, k = i3;

                        if (i == 0) {
                            uim = A0((j + 1) * h2, (k + 1) * h3);
                        } else if (i1 == 0) {
                            uim = preLeft[(r2 + i2) * r3 * Q3 + i3];
                        } else {
                            uim = U[((r1 + i1 - 1) * r2 + i2) * r3 * Q3 + i3];
                        }

                        if (i == Nx - 1) {
                            uip = A1((j + 1) * h2, (k + 1) * h3, X);
                        } else if (i2 == 0) {
                            uip = preBack[i1 * r3 * Q3 + i3];
                        } else {
                            uip = U[(i1 * r2 + i2 - 1) * r3 * Q3 + i3];
                        }

                        if (j == 0) {
                            ujm = B0((i + 1) * h1, (k + 1) * h3);
                        } else if (i2 == 0) {
                            ujm = preBack[(r1 + i1) * r3 * Q3 + i3];
                        } else {
                            ujm = U[((r1 + i1) * r2 + i2 - 1) * r3 * Q3 + i3];
                        }

                        if (j == Ny - 1) {
                            ujp = B1((i + 1) * h1, (k + 1) * h3, Y);
                        } else if (i1 == 0) {
                            ujp = preLeft[i2 * r3 * Q3 + i3];
                        } else {
                            ujp = U[((i1 - 1) * r2 + i2) * r3 * Q3 + i3];
                        }

                        if (k == 0) {
                            ukm = C0((i + 1) * h1, (j + 1) * h2);
                        } else {
                            ukm = U[((r1 + i1) * r2 + i2) * r3 * Q3 + i3 - 1];
                        }

                        if (k == Nz - 1) {
                            ukp = C1((i + 1) * h1, (j + 1) * h2, Z);
                        } else if (i1 == 0) {
                            if (i2 == 0) {
                                ukp = preBackLeft[i3 + 1];
                            } else {
                                ukp = preLeft[(i2 - 1) * r3 * Q3 + i3 + 1];
                            }
                        } else {
                            if (i2 == 0) {
                                ukp = preBack[(i1 - 1) * r3 * Q3 + i3 + 1];
                            } else {
                                ukp = U[((i1 - 1) * r2 + i2 - 1) * r3 * Q3 + i3 + 1];
                            }
                        }

                        if (i1 == 0) {
                            if (i2 == 0) {
                                u = preBackLeft[i3];
                            } else {
                                u = preLeft[(i2 - 1) * r3 * Q3 + i3];
                            }
                        } else {
                            if (i2 == 0) {
                                u = preBack[(i1  - 1) * r3 * Q3 + i3];
                            } else {
                                u = U[((i1 - 1) * r2 + i2 - 1) * r3 * Q3 + i3];
                            }
                        }

                        U[((r1 + i1) * r2 + i2) * r3 * Q3 + i3] =
                                w * ((uip + uim) / (h1 * h1)
                                     + (ujp + ujm) / (h2 * h2)
                                     + (ukp + ukm) / (h3 * h3) - F((i + 1) * h1,
                                                                   (j + 1) * h2,
                                                                   (k + 1) * h3)) /
                                (2 / (h1 * h1) + 2 / (h2 * h2) + 2 / (h3 * h3)) + (1 - w) * u;
                    }
                }
            }
//            printf("%d (%d, %d) leaves tile\n", rank, m, igl3);

            if (sendNeeded) {
                if (downRight != -1) {
//                    printf("%d (%d, %d) Isends downRight to %d \n", rank, m, igl3, downRight);
                    MPI_Send(U + ((2 * r1 * r2 - 1) * Q3 + igl3) * r3, r3, MPI_DOUBLE,
                             downRight, m * tag + igl3, grid_comm);
//                    printf("%d (%d, %d) Isent downRight to %d \n", rank, m, igl3, downRight);
                }

                if (down != -1) {
//                    printf("%d (%d, %d) Isends down to %d \n", rank, m, igl3, down);
                    MPI_Send(U + (((r1 + 1) * r2 - 1) * Q3 + igl3) * r3, 1, uik_t,
                              down, tag * m + igl3, grid_comm);
//                    printf("%d (%d, %d) Isent down to %d \n", rank, m, igl3, down);
                }

                if (right != - 1) {
//                    printf("%d (%d, %d) Isends right to %d \n", rank, m, igl3, right);
                    MPI_Send(U + ((2 * r1 - 1) * r2 * Q3 + igl3) * r3, 1, ujk_t,
                              right, tag * m + igl3, grid_comm);
//                    printf("%d (%d, %d) Isent right to %d \n", rank, m, igl3, right);
                }
            }
        }

        memcpy(U, U + r1 * r2 * r3 * Q3, r1 * r2 * r3 * Q3 * sizeof(double));
        memcpy(preLeft, preLeft + r2 * r3 * Q3, r2 * r3 * Q3 * sizeof(double));
        memcpy(preBack, preBack + r1 * r3 * Q3, r1 * r3 * Q3 * sizeof(double));
        memcpy(preBackLeft, preBackLeft + r3 * Q3, r3 * Q3 * sizeof(double));
    }

  
    double *R;
    if (rank == 0) {
        R = (double *)malloc(sizeof(double) * r1 * Q1 * r2 * Q2 * r3 * Q3);

        MPI_Datatype r_t;
        MPI_Type_vector(r1, r2 * r3 * Q3, r2 * Q2 * r3 * Q3, MPI_DOUBLE, &r_t);
        MPI_Type_commit(&r_t);

        int self[2];
        MPI_Cart_coords(grid_comm, rank, 2, self);
        for (int n = 0; n < dim[0]; ++n) {
            for (int m = 0; m < dim[1]; ++m) {
                if (n == self[0] && m == self[1]) {
                    for (int i = 0; i < r1; ++i) {
                        memcpy(R + ((n * r1 + i) * Q2 + m) * r2 * r3 * Q3,
                               U + i * r2 * r3 * Q3,
                               r2 * r3 * Q3 * sizeof(double));
                    }
                    continue;
                }
                int sender;
                int coords[2];
                coords[0] = n; coords[1] = m;
                MPI_Cart_rank(grid_comm, coords, &sender);
                MPI_Recv(R + (n * r1 * Q2 + m) * r2 * r3 * Q3, 1, r_t, sender, 10005000, grid_comm, &status);
            }
        }
    } else {
        MPI_Send(U, r1 * r2 * r3 * Q3, MPI_DOUBLE, 0, 10005000, grid_comm);
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
                    fprintf(f, "%f ", R[((rit + i) * r2 * Q2 + rit + j) * r3 * Q3 + k]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "-----------------------------------------------\n");
        }
        fclose(f);
    }

    return 0;
}