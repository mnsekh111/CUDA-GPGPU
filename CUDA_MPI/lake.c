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

double * extract_along_down(double *u, double *new, int x, int y, int n);
double * extract_along_side(double *u, double* new, int x, int y, int n);
void print_array(double *u, int n);

int taskId, totaltasks;

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

		h = (XMAX - XMIN) / npoints;

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

		print_heatmap("lake_i9.dat", u_i0, npoints, h);

		gettimeofday(&gpu_start, NULL);
		run_cpu9pt_mpi(u_gpu, u_i0, u_i1, pebs, npoints / 2, h, end_time);
		gettimeofday(&gpu_end, NULL);

		elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)
				- (gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
		printf("CPU took %f seconds\n", elapsed_gpu);

	} else {
		MPI_Recv(u_i0, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(u_i1, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(pebs, narea / totaltasks, MPI_DOUBLE, ROOT, DEFAULT_TAG,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if(taskId == 1)
			print_array(u_i0,npoints/2);

	}

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
	double *right_border, *left_border, *top_border, *down_border;

	un = (double*) malloc(sizeof(double) * n * n);
	uc = (double*) malloc(sizeof(double) * n * n);
	uo = (double*) malloc(sizeof(double) * n * n);

	right_border = (double*) malloc(sizeof(double) * n);
	left_border = (double*) malloc(sizeof(double) * n);
	top_border = (double*) malloc(sizeof(double) * n);
	down_border = (double*) malloc(sizeof(double) * n);

	memcpy(uo, u0, sizeof(double) * n * n);
	memcpy(uc, u1, sizeof(double) * n * n);

	t = 0.;
	dt = h / 2.;

	double cornor;
	while (1) {
		evolve9pt(un, uc, uo, pebbles, n, h, dt, t);
		switch (taskId) {
		case 0:
			extract_along_down(uc, right_border, 0, n - 1, n);
			extract_along_side(uc, down_border, n - 1, 0, n);
			cornor = uc[(n - 1) * n + (n - 1)];
			break;
		case 1:
			extract_along_down(uc, left_border, 0, 0, n);
			extract_along_side(uc, down_border, n - 1, 0,
					n);
			cornor = uc[(n - 1) * n + 0];
			//print_array(uc, n);
			break;
		case 2:
			extract_along_down(uc, right_border, 0, n - 1,
					n);
			extract_along_side(uc, top_border, 0, 0, n);
			cornor = uc[0 + n - 1];
			break;
		case 3:
			extract_along_down(uc, left_border, 0, 0, n);
			extract_along_side(uc, top_border, 0, 0, n);
			cornor = uc[0];
			break;

		}
		memcpy(uo, uc, sizeof(double) * n * n);
		memcpy(uc, un, sizeof(double) * n * n);

		if (!tpdt(&t, dt, end_time))
			break;
	}

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

double * extract_along_down(double *u, double*new, int x, int y, int n) {
	int i, index = 0;
	for (i = x; i < x + n; i++) {
		new[index++] = u[i * n + y];
	}
	return new;
}

double * extract_along_side(double *u, double *new, int x, int y, int n) {
	int j, index = 0;
	for (j = y; j < y + n; j++) {
		new[index++] = u[x * n + j];
	}
	return new;
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

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;

			if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
				un[idx] = 0.;
			} else {
				un[idx] = 2 * uc[idx] - uo[idx]
						+ VSQR * (dt * dt)
								* ((uc[idx - 1] + uc[idx + 1] + uc[idx - n]
										+ uc[idx + n]
										+ 0.25
												* (uc[idx - n - 1]
														+ uc[idx - n + 1]
														+ uc[idx + n - 1]
														+ uc[idx + n + 1])
										- 5 * uc[idx]) / (h * h)
										+ f(pebbles[idx], t));
			}
		}
	}
}

void print_heatmap(char *filename, double *u, int n, double h) {
	int i, j, idx;

	FILE *fp = fopen(filename, "w");

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;
			fprintf(fp, "%f %f %f\n", i * h, j * h, u[idx]);
		}
	}

	fclose(fp);
}
