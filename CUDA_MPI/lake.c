#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

#define ROOT 0
#define DEFAULT_TAG 0

void arr_div(double *, double *, int, int, int);
void init(double *u, double *pebbles, int n);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n,
		double h, double dt, double t);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n,
		double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(char *filename, double *u, int n, double h);
void init_pebbles(double *p, int pn, int n);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time);
void run_cpu9pt(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time);
void run_cpu9pt_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time);

extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time, int nthreads);

void extract_along_down(double *u, double *new, int x, int y, int n);
void extract_along_side(double *u, double* new, int x, int y, int n);
void print_array(double *u, int n);
void print_array1d(double *u, double n);

int taskId, totaltasks;
double *rec_right_border, *rec_left_border, *rec_top_border, *rec_down_border;
double rec_cornor;

int main(int argc, char *argv[]) {

	if (argc != 5) {
		printf("Usage: %s npoints npebs time_finish nthreads \n", argv[0]);
		return 0;
	}

	//Used for MPI
	int i, j;

	int npoints = atoi(argv[1]);
	int npebs = atoi(argv[2]);
	double end_time = (double) atof(argv[3]);
	int nthreads = atoi(argv[4]);
	int narea = npoints * npoints;

	double *u_i0, *u_i1;
	double *u_gpu, *pebs;
	double h;

	h = (XMAX - XMIN) / npoints;

	double elapsed_gpu;
	struct timeval gpu_start, gpu_end;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskId);
	MPI_Comm_size(MPI_COMM_WORLD, &totaltasks);

	MPI_Barrier(MPI_COMM_WORLD);
	u_i0 = (double*) malloc(sizeof(double) * narea / totaltasks);
	u_i1 = (double*) malloc(sizeof(double) * narea / totaltasks);
	pebs = (double*) malloc(sizeof(double) * narea / totaltasks);
	u_gpu = (double*) malloc(sizeof(double) * narea / totaltasks);

	if (taskId == ROOT) {
		printf("Running %s with (%d x %d) grid, until %f, with %d threads\n",
				argv[0], npoints, npoints, end_time, nthreads);

		double * global_u_i0 = (double*) malloc(sizeof(double) * narea);
		double * global_u_i1 = (double*) malloc(sizeof(double) * narea);
		double * global_pebs = (double*) malloc(sizeof(double) * narea);

		double * global_u_gpu = (double*) malloc(sizeof(double) * narea);

		init_pebbles(global_pebs, npebs, npoints);
		init(global_u_i0, global_pebs, npoints);
		init(global_u_i1, global_pebs, npoints);

		//print_array(global_u_i0,npoints);
		arr_div(u_i0, global_u_i0, 0, npoints / 2, npoints);
		arr_div(u_i1, global_u_i1, 0, npoints / 2, npoints);
		arr_div(pebs, global_pebs, 0, npoints / 2, npoints);
		MPI_Send(u_i0, narea / totaltasks, MPI_DOUBLE, 1, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(u_i1, narea / totaltasks, MPI_DOUBLE, 1, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(pebs, narea / totaltasks, MPI_DOUBLE, 1, DEFAULT_TAG,
		MPI_COMM_WORLD);

		arr_div(u_i0, global_u_i0, npoints / 2, 0, npoints);
		arr_div(u_i1, global_u_i1, npoints / 2, 0, npoints);
		arr_div(pebs, global_pebs, npoints / 2, 0, npoints);
		MPI_Send(u_i0, narea / totaltasks, MPI_DOUBLE, 2, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(u_i1, narea / totaltasks, MPI_DOUBLE, 2, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(pebs, narea / totaltasks, MPI_DOUBLE, 2, DEFAULT_TAG,
		MPI_COMM_WORLD);

		arr_div(u_i0, global_u_i0, npoints / 2, npoints / 2, npoints);
		arr_div(u_i1, global_u_i1, npoints / 2, npoints / 2, npoints);
		arr_div(pebs, global_pebs, npoints / 2, npoints / 2, npoints);

		MPI_Send(u_i0, narea / totaltasks, MPI_DOUBLE, 3, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(u_i1, narea / totaltasks, MPI_DOUBLE, 3, DEFAULT_TAG,
		MPI_COMM_WORLD);
		MPI_Send(pebs, narea / totaltasks, MPI_DOUBLE, 3, DEFAULT_TAG,
		MPI_COMM_WORLD);

		arr_div(u_i0, global_u_i0, 0, 0, npoints);
		arr_div(u_i1, global_u_i1, 0, 0, npoints);
		arr_div(pebs, global_pebs, 0, 0, npoints);

		//print_array(u_i0, npoints / 2);

		run_cpu9pt_mpi(u_gpu, u_i0, u_i1, pebs, npoints / 2, h, end_time);
		print_heatmap("lake_f9_0.dat", u_gpu, npoints / 2, h);

	} else {
		MPI_Recv(u_i0, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(u_i1, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(pebs, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		run_cpu9pt_mpi(u_gpu, u_i0, u_i1, pebs, npoints / 2, h, end_time);

		char fname[20];
		snprintf(fname, 20, "lake_f9_%d.dat", taskId);
		print_heatmap(fname, u_gpu, npoints / 2, h);

	}

	MPI_Barrier(MPI_COMM_WORLD);
	free(u_i0);
	free(u_i1);
	free(pebs);
	free(u_gpu);

	MPI_Finalize();
	return 0;
}

void print_array(double *u, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			printf("%lf ", u[i * n + j]);
		printf("\n");
	}
}

void print_array1d(double *u, double n) {
	int i;
	for (i = 0; i < n; i++)
		printf("%lf ", u[i]);
	printf("\n");
}

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time) {
	double *un, *uc, *uo;
	double t, dt;

	un = (double*) malloc(sizeof(double) * n * n);
	uc = (double*) malloc(sizeof(double) * n * n);
	uo = (double*) malloc(sizeof(double) * n * n);

	memcpy(uo, u0, sizeof(double) * n * n);
	memcpy(uc, u1, sizeof(double) * n * n);

	t = 0.;
	dt = h / 2.;

	while (1) {
		evolve(un, uc, uo, pebbles, n, h, dt, t);

		memcpy(uo, uc, sizeof(double) * n * n);
		memcpy(uc, un, sizeof(double) * n * n);

		if (!tpdt(&t, dt, end_time))
			break;
	}

	memcpy(u, un, sizeof(double) * n * n);
}

void run_cpu9pt_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time) {
	double *un, *uc, *uo;
	double t, dt;
	double *send_right_border, *send_left_border, *send_top_border,
			*send_down_border;

	double send_cornor;

	un = (double*) malloc(sizeof(double) * n * n);
	uc = (double*) malloc(sizeof(double) * n * n);
	uo = (double*) malloc(sizeof(double) * n * n);

	send_right_border = (double*) malloc(sizeof(double) * n);
	send_left_border = (double*) malloc(sizeof(double) * n);
	send_top_border = (double*) malloc(sizeof(double) * n);
	send_down_border = (double*) malloc(sizeof(double) * n);

	rec_right_border = (double*) malloc(sizeof(double) * n);
	rec_left_border = (double*) malloc(sizeof(double) * n);
	rec_top_border = (double*) malloc(sizeof(double) * n);
	rec_down_border = (double*) malloc(sizeof(double) * n);

	MPI_Request reqs[6];
	MPI_Status stats[6];

	memcpy(uo, u0, sizeof(double) * n * n);
	memcpy(uc, u1, sizeof(double) * n * n);

	t = 0.;
	dt = h / 2.;

	int cnt = 0;

	//Testing for 1 iteration
	while (cnt++ < 2) {

		switch (taskId) {
		case 0:

			MPI_Irecv(rec_right_border, n, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(rec_down_border, n, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[1]);
			MPI_Irecv(&rec_cornor, 1, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[2]);

			extract_along_down(uc, send_right_border, 0, n - 1, n);
			extract_along_side(uc, send_down_border, n - 1, 0, n);
			send_cornor = uc[(n - 1) * n + (n - 1)];

			MPI_Isend(send_right_border, n, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[3]);
			MPI_Isend(send_down_border, n, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[4]);
			MPI_Isend(&send_cornor, 1, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[5]);
			break;

		case 1:

			MPI_Irecv(rec_left_border, n, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(rec_down_border, n, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[1]);
			MPI_Irecv(&rec_cornor, 1, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[2]);

			extract_along_down(uc, send_left_border, 0, 0, n);
			extract_along_side(uc, send_down_border, n - 1, 0, n);
			send_cornor = uc[(n - 1) * n + 0];

			MPI_Isend(send_left_border, n, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[3]);
			MPI_Isend(send_down_border, n, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[4]);
			MPI_Isend(&send_cornor, 1, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[5]);

			//print_array1d(send_left_border, n);
			//print_array(uc, n);
			break;
		case 2:

			MPI_Irecv(rec_right_border, n, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(rec_top_border, n, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[1]);
			MPI_Irecv(&rec_cornor, 1, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[2]);

			extract_along_down(uc, send_right_border, 0, n - 1, n);
			extract_along_side(uc, send_top_border, 0, 0, n);
			send_cornor = uc[0 + n - 1];

			MPI_Isend(send_right_border, n, MPI_DOUBLE, 3, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[3]);
			MPI_Isend(send_top_border, n, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[4]);
			MPI_Isend(&send_cornor, 1, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[5]);
			break;
		case 3:

			MPI_Irecv(rec_left_border, n, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(rec_top_border, n, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[1]);
			MPI_Irecv(&rec_cornor, 1, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[2]);

			extract_along_down(uc, send_left_border, 0, 0, n);
			extract_along_side(uc, send_top_border, 0, 0, n);
			send_cornor = uc[0];

			MPI_Isend(send_left_border, n, MPI_DOUBLE, 2, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[3]);
			MPI_Isend(send_top_border, n, MPI_DOUBLE, 1, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[4]);
			MPI_Isend(&send_cornor, 1, MPI_DOUBLE, 0, DEFAULT_TAG,
			MPI_COMM_WORLD, &reqs[5]);
			break;

		}

		MPI_Waitall(6, reqs, stats);
		MPI_Barrier(MPI_COMM_WORLD);

		evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

//		int b=2;
//		if (taskId == 0 && cnt == b) {
//			print_array1d(rec_right_border, n);
//			print_array1d(rec_down_border, n);
//			printf("\n-------------\n");
//		}
//
//		if (taskId == 1 && cnt == b) {
//				print_array1d(rec_left_border, n);
//				print_array1d(rec_down_border, n);
//				printf("\n-------------\n");
//			}
//
//		if (taskId == 2 && cnt == b) {
//				print_array1d(rec_right_border, n);
//				print_array1d(rec_top_border, n);
//				printf("\n-------------\n");
//			}
//
//		if (taskId == 3 && cnt == b) {
//				print_array1d(rec_left_border, n);
//				print_array1d(rec_top_border, n);
//				printf("\n-------------\n");
//			}

//		if(cnt < 5)
//			if(taskId == 0)
//				print_array(un,n);
//		if (taskId == 3) {
//			//print_array1d(rec_top_border, n);
//			printf("\n------------\n");
//			print_array(un, n);
//		}
//		if (cnt++ == 1)
//			print_array(uo, n);
//		evolve9pt(un, uc, uo, pebbles, n, h, dt, t);
//		if (cnt++ == 2) {
//			printf("\n-------\n");
//			print_array(un, n);
//		}

		memcpy(uo, uc, sizeof(double) * n * n);
		memcpy(uc, un, sizeof(double) * n * n);

		if (!tpdt(&t, dt, end_time))
			break;
	}

//	if(taskId == 0)
//		print_array(un,n);
	memcpy(u, un, sizeof(double) * n * n);

}

void run_cpu9pt(double *u, double *u0, double *u1, double *pebbles, int n,
		double h, double end_time) {
	double *un, *uc, *uo;
	double t, dt;

	un = (double*) malloc(sizeof(double) * n * n);
	uc = (double*) malloc(sizeof(double) * n * n);
	uo = (double*) malloc(sizeof(double) * n * n);

	memcpy(uo, u0, sizeof(double) * n * n);
	memcpy(uc, u1, sizeof(double) * n * n);

	t = 0.;
	dt = h / 2.;

	while (1) {
		evolve9pt(un, uc, uo, pebbles, n, h, dt, t);

		memcpy(uo, uc, sizeof(double) * n * n);
		memcpy(uc, un, sizeof(double) * n * n);

		if (!tpdt(&t, dt, end_time))
			break;
	}

	memcpy(u, un, sizeof(double) * n * n);
}

void extract_along_down(double *u, double* new, int x, int y, int n) {
	int i, index = 0;
	for (i = x; i < x + n; i++) {
		new[index++] = u[i * n + y];
	}
}

void extract_along_side(double *u, double *new, int x, int y, int n) {
	int j, index = 0;
	for (j = y; j < y + n; j++) {
		new[index++] = u[x * n + j];
	}
}

void arr_div(double *u, double *global, int x, int y, int n) {
	int index = 0, i, j;
	for (i = x; i < x + n / 2; i++)
		for (j = y; j < y + n / 2; j++) {
			u[index++] = global[i * n + j];
		}
}

void init_pebbles(double *p, int pn, int n) {
	int i, j, k, idx;
	int sz;

	srand(time(NULL));
	memset(p, 0, sizeof(double) * n * n);

	for (k = 0; k < pn; k++) {
		i = rand() % (n - 4) + 2;
		j = rand() % (n - 4) + 2;
		sz = rand() % MAX_PSZ;
		idx = j + i * n;
		p[idx] = (double) sz;
	}

}

double f(double p, double t) {
	return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf) {
	if ((*t) + dt > tf)
		return 0;
	(*t) = (*t) + dt;
	return 1;
}

void init(double *u, double *pebbles, int n) {
	int i, j, idx;

	int index = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;
//			u[idx] = f(pebbles[idx], 0.0);
			u[idx] = index++;
		}
	}
}

void evolve(double *un, double *uc, double *uo, double *pebbles, int n,
		double h, double dt, double t) {
	int i, j, idx;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;

			if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
				un[idx] = 0.;
			} else {
				un[idx] = 2 * uc[idx] - uo[idx]
						+ VSQR * (dt * dt)
								* ((uc[idx - 1] + uc[idx + 1] + uc[idx + n]
										+ uc[idx - n] - 4 * uc[idx]) / (h * h)
										+ f(pebbles[idx], t));
			}
		}
	}
}

/*
 *Implementation of evolve9pt method that performs 9-point stencil under CPU.
 */
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n,
		double h, double dt, double t) {
	int i, j, idx;
	double L, R, T, B, LT, RT, LB, RB;

	static int cnt = 0;
	cnt++;
	int b = 1;
//	if (taskId == 0 && cnt == b) {
//		print_array1d(rec_right_border, n);
//		print_array1d(rec_down_border, n);
//		printf("\n-------------\n");
//	}
//
//	if (taskId == 1 && cnt == b) {
//			print_array1d(rec_left_border, n);
//			print_array1d(rec_down_border, n);
//			printf("\n-------------\n");
//		}
//
//	if (taskId == 2 && cnt == b) {
//			print_array1d(rec_right_border, n);
//			print_array1d(rec_top_border, n);
//			printf("\n-------------\n");
//		}
//
//	if (taskId == 3 && cnt == b) {
//			print_array1d(rec_left_border, n);
//			print_array1d(rec_top_border, n);
//			printf("\n-------------\n");
//		}

	switch (taskId) {
	case 0:
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				idx = j + i * n;

				if (i == 0 || j == 0) {
					un[idx] = 0.;

				} else {
					if (i == n - 1 && j == n - 1) {

						L = uc[idx - 1];
						R = rec_right_border[i];
						T = uc[idx - n];
						B = rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = rec_right_border[i - 1];
						LB = rec_down_border[j - 1];
						RB = rec_cornor;
					} else if (i == n - 1) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = rec_down_border[j - 1];
						RB = rec_down_border[j + 1];
					} else if (j == n - 1) {
						L = uc[idx - 1];
						R = rec_right_border[i];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = rec_right_border[i - 1];
						LB = uc[idx + n - 1];
						RB = rec_right_border[i + 1];
					} else {

						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					}

					un[idx] = 2 * uc[idx] - uo[idx]
							+ VSQR * (dt * dt)
									* ((L + R + T + B
											+ 0.25 * (LT + RT + LB + RB)
											- 5 * uc[idx]) / (h * h)
											+ f(pebbles[idx], t));

//					un[idx] = L + R + T + B + LT + RT + LB + RB;
				}

			}
		}
		break;
	case 1:
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				idx = j + i * n;

				if (i == 0 || j == n - 1) {
					un[idx] = 0.;

				} else {
					if (i == n - 1 && j == 0) {
						L = rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = rec_down_border[j];
						LT = rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = rec_cornor;
						RB = rec_down_border[j + 1];
					} else if (i == n - 1) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = rec_down_border[j - 1];
						RB = rec_down_border[j + 1];
					} else if (j == 0) {
						L = rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = rec_left_border[i + 1];
						RB = uc[idx + n + 1];
					} else {

						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];

						//if(cnt == 1)
						//printf("%f %f %f %f %f %f %f %f", L,R,T,B,LT,RT,LB,RB);

					}

					un[idx] = 2 * uc[idx] - uo[idx]
							+ VSQR * (dt * dt)
									* ((L + R + T + B
											+ 0.25 * (LT + RT + LB + RB)
											- 5 * uc[idx]) / (h * h)
											+ f(pebbles[idx], t));


					if (cnt == 1)
						printf("%f", un[idx]);
					//printf("%f ", un[idx]);
					//un[idx] = L + R + T + B + LT + RT + LB + RB;
				}

			}
		}
		break;
	case 2:
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				idx = j + i * n;

				if (i == n - 1 || j == 0) {
					un[idx] = 0.;

				} else {
					if (i == 0 && j == n - 1) {

						L = uc[idx - 1];
						R = rec_right_border[i];
						T = rec_top_border[j];
						B = uc[idx + n];
						LT = rec_top_border[j - 1];
						RT = rec_cornor;
						LB = uc[idx + n - 1];
						RB = rec_right_border[i + 1];
					} else if (i == 0) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = rec_top_border[j];
						B = uc[idx + n];
						LT = rec_top_border[j - 1];
						RT = rec_top_border[j + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					} else if (j == n - 1) {
						L = uc[idx - 1];
						R = rec_right_border[i];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = rec_right_border[i - 1];
						LB = uc[idx + n - 1];
						RB = rec_right_border[i + 1];
					} else {

						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					}

					un[idx] = 2 * uc[idx] - uo[idx]
							+ VSQR * (dt * dt)
									* ((L + R + T + B
											+ 0.25 * (LT + RT + LB + RB)
											- 5 * uc[idx]) / (h * h)
											+ f(pebbles[idx], t));
//					un[idx] = L + R + T + B + LT + RT + LB + RB;
				}

			}
		}

		break;
	case 3:
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				idx = j + i * n;

				if (i == n - 1 || j == n - 1) {
					un[idx] = 0.;

				} else {
					if (i == 0 && j == 0) {
						L = rec_left_border[i];
						R = uc[idx + 1];
						T = rec_top_border[j];
						B = uc[idx + n];
						LT = rec_cornor;
						RT = rec_top_border[j + 1];
						LB = rec_left_border[i + 1];
						RB = uc[idx + n + 1];
					} else if (i == 0) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = rec_top_border[j];
						B = uc[idx + n];
						LT = rec_top_border[j - 1];
						RT = rec_top_border[j + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					} else if (j == 0) {
						L = rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = rec_left_border[i + 1];
						RB = uc[idx + n + 1];
					} else {

						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					}

					un[idx] = 2 * uc[idx] - uo[idx]
							+ VSQR * (dt * dt)
									* ((L + R + T + B
											+ 0.25 * (LT + RT + LB + RB)
											- 5 * uc[idx]) / (h * h)
											+ f(pebbles[idx], t));
					//un[idx] = L + R + T + B + LT + RT + LB + RB;
				}

			}
		}
		break;
	}

//	if(cnt == b){
//		print_array(un,n);
//		printf("----------------\n");
//	}

}

void print_heatmap(char *filename, double *u, int n, double h) {
	int i, j, idx;

	FILE *fp = fopen(filename, "w");

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;
			fprintf(fp, "%f %f %e\n", i * h, j * h, u[idx]);
		}
	}

	fclose(fp);
}
