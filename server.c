/*
 * server.c
 *
 *  Created on: Feb 6, 2016
 *      Author: mns
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

int sendMessages();

int main(int argc, char *argv[]) {

	/* Variables used by server socket*/
	int ssockFd;
	int ccsockFd;
	int iSetOption = 1;
	struct sockaddr_in server;
	char buff[1024]; // 1 KB Buffer
	int returnVal;

	/* Socket creation*/
	ssockFd = socket(AF_INET, SOCK_STREAM, 0);
	setsockopt(ssockFd, SOL_SOCKET, SO_REUSEADDR, (char*) &iSetOption,
			sizeof(iSetOption));

	if (ssockFd < 0) {
		perror("Cannot create server socket\n");
		exit(1);
	}

	server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons(5000);

	/* call bind*/
	if (bind(ssockFd, (struct sockaddr*) &server, sizeof(server))) {
		perror("Unable to bind socket\n");
		exit(1);
	}

	/* Listen for connections from client*/
	listen(ssockFd, 10);

	do {
		ccsockFd = accept(ssockFd, (struct sockaddr *) 0, 0);
		if (ccsockFd == -1) {
			perror("Could not accept connection from the client\n");
			close(ccsockFd);
		} else {
			memset(buff, 0, sizeof(buff));
			returnVal = recv(ccsockFd, buff, sizeof(buff), 0);
			if (returnVal < 0) {
				perror("Receive error. Unable to read stream\n");
			} else if (returnVal == 0) {
				printf("Connection with client ended \n");
			} else {
				printf("Received : %s\n", buff);
				sendMessages(ccsockFd);
			}
			close(ccsockFd);
		}
	} while (1);

	/* Accept the connection from client */
	return 0;
}

int sendMessages(int destFd) {
	int MSG_SIZES = 17;
	int REPS = 10;
	int i, k;
	int msg_size = 32;
	int returnVal = 0;
	char * dummy_rec_data, *dummy_send_data;

	for (i = 0; i < MSG_SIZES; i++, msg_size *= 2) {
		dummy_send_data = (char *) malloc(msg_size);
		dummy_rec_data = (char *) malloc(msg_size);

		memset(dummy_send_data, 'A', msg_size - 1);
		dummy_send_data[msg_size - 1] = '\0';

		for (k = 0; k < REPS; k++) {
			send(destFd, dummy_send_data, msg_size, 0);
			returnVal = recv(destFd, dummy_rec_data, msg_size, 0);
			printf("Received from client %d on rep %d\n", msg_size, k);
		}

	}
}
