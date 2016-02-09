/*
 * custom_mpi.h
 *
 *  Created on: Feb 6, 2016
 *      Author: mns
 */

#include "mpi.h"

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
		MPI_Comm comm, MPI_Status *status) MPICH_ATTR_POINTER_WITH_TYPE_TAG(1,3);

int MPI_Barrier(MPI_Comm comm);
