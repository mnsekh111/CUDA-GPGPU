#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

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
}
			close(ccsockFd);
		}
	} while (1);

	/* Accept the connection from client */
	return 0;
}
