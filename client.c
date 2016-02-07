/*
 * client.c
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
#include <netdb.h>
#include <netinet/in.h>

#define DATA "Hello from client"

int receiveMessages(int);

int main(int argc, char *argv[]) {

	/* Variables used by server socket*/
	int sockFd;
	struct sockaddr_in server;
	struct hostent *host;
	char buff[1024]; // 1 KB Buffer

	/* Socket creation*/
	sockFd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockFd < 0) {
		perror("Cannot create client socket\n");
		exit(1);
	}

	server.sin_family = AF_INET;
	host = gethostbyname(argv[1]);

	if (host == 0) {
		perror("Unable to get host gethostbyname failed\n");
		close(sockFd);
		exit(1);
	}

	memcpy(&server.sin_addr, host->h_addr,host->h_length);
	server.sin_port = htons(5000);

	if (connect(sockFd, (struct sockaddr*) &server, sizeof(server)) < 0) {
		perror("Unable to connect to the server");
		close(sockFd);
		exit(1);
	}


	if (send(sockFd, DATA, sizeof(DATA), 0) < 0) {
		perror("send failed");
		close(sockFd);
		exit(1);
	}

	printf("Sent data successfully to server \n");
	return 0;
}
