#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG
#define VSQR 0.1
#define TSCALE 1.0
#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)


extern int taskId, totaltasks;
extern double *rec_right_border, *rec_left_border, *rec_top_border, *rec_down_border;
extern double rec_cornor;
extern "C" int tpdt(double *t, double dt, double end_time);
extern "C" void do_transfer(double* uc,int n);

/**************************************
 * void __cudaSafeCall(cudaError err, const char *file, const int line)
 * void __cudaCheckError(const char *file, const int line)
 *
 * These routines were taken from the GPU Computing SDK
 * (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
 **************************************/
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
        do {
                if (cudaSuccess != err) {
                        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file,
                                        line, cudaGetErrorString(err));
                        exit(-1);
                }
        } while (0);
#pragma warning( pop )
#endif  // __DEBUG
        return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
        do {
                cudaError_t err = cudaGetLastError();
                if (cudaSuccess != err) {
                        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n", file,
                                        line, cudaGetErrorString(err));
                        exit(-1);
                }
                // More careful checking. However, this will affect performance.
                // Comment if not needed.
                /*err = cudaThreadSynchronize();
                 if( cudaSuccess != err )
                 {
                 fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
                 file, line, cudaGetErrorString( err ) );
                 exit( -1 );
                 }*/
        } while (0);
#pragma warning( pop )
#endif // __DEBUG
        return;
}

__device__ double f_gpu(double p, double t) {
        return -__expf(-TSCALE * t) * p;
}

__device__ int tpdt_gpu(double *t, double dt, double tf) {
        if ((*t) + dt > tf)
                return 0;
        (*t) = (*t) + dt;
        return 1;
}

/*
This is the kernel that performs a quadrant of work assigned to each task
*/
__global__ void evolve9ptgpu(double *un, double *uc, double *uo,
                double *pebbles, int n, double h, double dt, double t,
                double end_time,int taskId,double rec_cornor,double *g_rec_down_border,double *g_rec_right_border,double *g_rec_left_border,double *g_rec_top_border) {

        int gridOffset = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x
                        * blockDim.y;
        int blockOffset = threadIdx.x * blockDim.x + threadIdx.y;
        int idx = gridOffset + blockOffset;
        int i, j;
        float f = -expf(-1.0 * t) * pebbles[idx];
	double L, R, T, B, LT, RT, LB, RB;

	// Do different operations for different task id.
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
						R = g_rec_right_border[i];
						T = uc[idx - n];
						B = g_rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = g_rec_right_border[i - 1];
						LB = g_rec_down_border[j - 1];
						RB = rec_cornor;
					} else if (i == n - 1) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = g_rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = g_rec_down_border[j - 1];
						RB = g_rec_down_border[j + 1];
					} else if (j == n - 1) {
						L = uc[idx - 1];
						R = g_rec_right_border[i];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = g_rec_right_border[i - 1];
						LB = uc[idx + n - 1];
						RB = g_rec_right_border[i + 1];
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
											+ f);

//                                      un[idx] = L + R + T + B + LT + RT + LB + RB;
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
						L = g_rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = g_rec_down_border[j];
						LT = g_rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = rec_cornor;
						RB = g_rec_down_border[j + 1];
					} else if (i == n - 1) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = g_rec_down_border[j];
						LT = uc[idx - n - 1];
						RT = uc[idx - n + 1];
						LB = g_rec_down_border[j - 1];
						RB = g_rec_down_border[j + 1];
					} else if (j == 0) {
						L = g_rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = g_rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = g_rec_left_border[i + 1];
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
											+ f);
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
						R =g_rec_right_border[i];
						T = g_rec_top_border[j];
						B = uc[idx + n];
						LT = g_rec_top_border[j - 1];
						RT = rec_cornor;
						LB = uc[idx + n - 1];
						RB = g_rec_right_border[i + 1];
					} else if (i == 0) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = g_rec_top_border[j];
						B = uc[idx + n];
						LT = g_rec_top_border[j - 1];
						RT = g_rec_top_border[j + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					} else if (j == n - 1) {
						L = uc[idx - 1];
						R = g_rec_right_border[i];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = uc[idx - n - 1];
						RT = g_rec_right_border[i - 1];
						LB = uc[idx + n - 1];
						RB = g_rec_right_border[i + 1];
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
											+ f);
//                                      un[idx] = L + R + T + B + LT + RT + LB + RB;
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
						L = g_rec_left_border[i];
						R = uc[idx + 1];
						T = g_rec_top_border[j];
						B = uc[idx + n];
						LT = rec_cornor;
						RT = g_rec_top_border[j + 1];
						LB = g_rec_left_border[i + 1];
						RB = uc[idx + n + 1];
					} else if (i == 0) {
						L = uc[idx - 1];
						R = uc[idx + 1];
						T = g_rec_top_border[j];
						B = uc[idx + n];
						LT = g_rec_top_border[j - 1];
						RT = g_rec_top_border[j + 1];
						LB = uc[idx + n - 1];
						RB = uc[idx + n + 1];
					} else if (j == 0) {
						L = g_rec_left_border[i];
						R = uc[idx + 1];
						T = uc[idx - n];
						B = uc[idx + n];
						LT = g_rec_left_border[i - 1];
						RT = uc[idx - n + 1];
						LB = g_rec_left_border[i + 1];
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
							* ((L + R + T + B + 0.25 * (LT + RT + LB + RB)
									- 5 * uc[idx]) / (h * h)
									+ f);
					//un[idx] = L + R + T + B + LT + RT + LB + RB;
				}

			}
		}
		break;
	}


}

extern "C" void run_gpu9pt_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
                double h, double end_time, int nthreads) {
        cudaEvent_t kstart, kstop;
        float ktime;

        double *un, *uc, *uo, *pb;
        double t, dt;

        double *uccpu = (double*) malloc(sizeof(double) * n * n);
		double * g_rec_left_border;
		double * g_rec_top_border;
		double * g_rec_down_border;
		double * g_rec_right_border;

        /* Set up device timers */
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaEventCreate(&kstart));
        CUDA_CALL(cudaEventCreate(&kstop));

        cudaMalloc((void **) &un, sizeof(double) * n * n);
        cudaMalloc((void **) &uc, sizeof(double) * n * n);
        cudaMalloc((void **) &uo, sizeof(double) * n * n);
        cudaMalloc((void **) &pb, sizeof(double) * n * n);

        dim3 block_dim(nthreads, nthreads, 1);
        dim3 grid_dim(n / nthreads, n / nthreads, 1);

        cudaMemcpy(uo, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(uc, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(pb, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);
        CUDA_CALL(cudaMemcpy(uccpu, uc, sizeof(double) * n * n,
                                                cudaMemcpyDeviceToHost));

        t = 0.;
        dt = h / 2.;

        /* Start GPU computation timer */
        CUDA_CALL(cudaEventRecord(kstart, 0));

		int cnt = 0;
		
		/**
			The loop is required to run only for 1 time in the
			question
		*/
        while (cnt++ < 1) {

                do_transfer(uccpu,n);
                cudaMemcpy(g_rec_left_border, rec_left_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_top_border, rec_top_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_right_border, rec_right_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_down_border, rec_down_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);



                evolve9ptgpu<<<grid_dim, block_dim>>>(un, uc, uo, pb, n/4, h, dt, t,
                                end_time,taskId,rec_cornor,g_rec_down_border,g_rec_right_border,g_rec_left_border,g_rec_top_border);

                CUDA_CALL(
                                cudaMemcpy(uo, uc, sizeof(double) * n * n,
                                                cudaMemcpyDeviceToDevice));
                CUDA_CALL(
                                cudaMemcpy(uc, un, sizeof(double) * n * n,
                                                cudaMemcpyDeviceToDevice));


                cudaMemcpy(rec_left_border, g_rec_left_border, sizeof(double) * n,
                                cudaMemcpyDeviceToHost);
                cudaMemcpy(rec_top_border, g_rec_top_border, sizeof(double) * n,
                                cudaMemcpyDeviceToHost);
                cudaMemcpy(rec_right_border, g_rec_right_border, sizeof(double) * n,
                                cudaMemcpyDeviceToHost);
                cudaMemcpy(rec_down_border, g_rec_down_border, sizeof(double) * n,
                                cudaMemcpyDeviceToHost);

                if (!tpdt(&t, dt, end_time))
                        break;
        }

        cudaMemcpy(u, un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
        /* Stop GPU computation timer */
        CUDA_CALL(cudaEventRecord(kstop, 0));
        CUDA_CALL(cudaEventSynchronize(kstop));
        CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
        printf("GPU computation: %f msec\n", ktime);

        cudaFree(un);
        cudaFree(uc);
        cudaFree(uo);
        cudaFree(pb);

        /* timer cleanup */
        CUDA_CALL(cudaEventDestroy(kstart));
        CUDA_CALL(cudaEventDestroy(kstop));
}
