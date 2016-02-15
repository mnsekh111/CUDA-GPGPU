#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG
#define VSQR 0.1
#define TSCALE 1.0
#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

 double * g_rec_left_border;
 double * g_rec_top_border;
 double * g_rec_down_border;
 double * g_rec_right_border;

extern int taskId, totaltasks;
extern double *rec_right_border, *rec_left_border, *rec_top_border, *rec_down_border;
extern double rec_cornor;
extern int tpdt(double *t, double dt, double end_time);
extern void do_transfer(double* uc,int n);

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

__global__ void evolve9ptgpu(double *un, double *uc, double *uo,
                double *pebbles, int n, double h, double dt, double t,
                double end_time) {

        int gridOffset = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x
                        * blockDim.y;
        int blockOffset = threadIdx.x * blockDim.x + threadIdx.y;
        int idx = gridOffset + blockOffset;

        if ((blockIdx.x == 0 && threadIdx.x == 0)
                        || (blockIdx.y == 0 && threadIdx.y == 0)
                        || (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1)
                        || (blockIdx.y == gridDim.y - 1 && threadIdx.y == blockDim.y - 1)) {
                un[idx] = 0.;
        } else {
                un[idx] =
                                2 * uc[idx] - uo[idx]
                                                + VSQR * (dt * dt)
                                                                * ((uc[idx - 1] + uc[idx + 1] + uc[idx - n]
                                                                        + uc[idx + n]
                                                                        + 0.25
                                                                        * (uc[idx - n - 1]
                                                                        + uc[idx - n + 1]
                                                                        + uc[idx + n - 1]
                                                                        + uc[idx + n + 1])
                                                                        - 5 * uc[idx]) / (h * h)
                                                                        + f_gpu(pebbles[idx], t));
        }

}

extern "C" void run_gpu9pt_mpi(double *u, double *u0, double *u1, double *pebbles, int n,
                double h, double end_time, int nthreads) {
        cudaEvent_t kstart, kstop;
        float ktime;

        double *un, *uc, *uo, *pb;
        double t, dt;

        double *uccpu = (double*) malloc(sizeof(double) * n * n);


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
        while (cnt++<1) {

                //do_transfer(uccpu,n);
                cudaMemcpy(g_rec_left_border, rec_left_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_top_border, rec_top_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_right_border, rec_right_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);
                cudaMemcpy(g_rec_down_border, rec_down_border, sizeof(double) * n,
                                cudaMemcpyHostToDevice);


                evolve9ptgpu<<<grid_dim, block_dim>>>(un, uc, uo, pb, n, h, dt, t,
                                end_time);

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

                /*if (!tpdt(&t, dt, end_time))
                        break; */
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

