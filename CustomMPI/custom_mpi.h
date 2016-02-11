/*
 * custom_mpi.h
 *
 *  Created on: Feb 06, 2016
 *      Author: mns
 */

/*
 * The functions that need to be over ridden from mpi.h
 */

typedef struct MPI_Status {
	int MPI_SOURCE;
	int MPI_TAG;
	int MPI_ERROR;
	MPI_Count count;
	int cancelled;
	int abi_slush_fund[2];

} MPI_Status;

typedef struct comm_env {
	int comm_size;
	int comm_rank;
	char ** processor_names;
} MPI_Comm;

MPI_Comm MPI_COMM_WORLD;

#define MPI_SUCCESS          0      /* Successful return code */
#define MPI_STATUS_IGNORE (MPI_Status *)1

#define MPI_CHAR 1
#define MPI_DOUBLE 8

int MPI_Init(int *argc, char **argv[]);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Get_processor_name(char *name, int *resultlen);

int MPI_Send(void *buf, int count, int datatype, int dest, int tag,
		MPI_Comm comm);
int MPI_Recv(void *buf, int count, int datatype, int source, int tag,
		MPI_Comm comm, MPI_Status *status);
int MPI_Barrier(MPI_Comm comm);
double MPI_Wtime(void);
int MPI_Finalize(void);

