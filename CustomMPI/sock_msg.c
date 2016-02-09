/*
 * sock_msg.c
 *
 *  Created on: Feb 6, 2016
 *      Author: mns
 */

/************************************************************************/
/*                                                                      */
/* Copyright (C) 1996 by Frank Mueller.                                 */
/* This code was developed at Humboldt University Berlin.               */
/* Distributed by the author(s) under the following terms:              */
/*                                                                      */
/* This code may not be distributed further without permission          */
/*      from Frank Mueller of the Humboldt-University zu Berlin,        */
/*      mueller@informatik.hu-berlin.de or (+49) (30) 20181-276.        */
/* This code is distributed "AS IS" WITHOUT ANY WARRANTY; without even  */
/*      even the implied warranty of MERCHANTABILITY OR FITNESS FOR A   */
/*      PARTICULAR PURPOSE.  No claims are made as to whether it serves */
/*      any particular purpose or even works at all.                    */
/*                                                                      */
/************************************************************************/

/*
 * This file is part of the distributed shared memory threads
 *
 *   $Id: sock_msg.c,v 1.23 2001/01/12 14:24:42 roeblitz Exp $
 *
 * DSM socket communication layer, comm. type = {SOCK_STREAM, SOCK_DGRAM}
 */

#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#define DSM_LINGER SO_LINGER

/*
 * dsm_server_open - open a communication medium of type,
 * either use fixed port or fill in port (if initially 0),
 * optionally for broadcast (if TRUE),
 * optionally fill in a medium address gaddr (if gaddr != NULL)
 * optionally set send and receive buffer size to bufsize (unless 0)
 * return a handle "s" (used for accept)
 */
int dsm_server_open(int type, int *port, int broadcast,
		struct sockaddr_in *gaddr, int bufsize) {
	int s;
	struct sockaddr_in from, name;
	struct netent *net;
	struct hostent *hp;
	int prot, len;
	int on = 1;
	struct linger l;
	struct sockaddr_in laddr, *saddr;

	saddr = (gaddr != NULL ? gaddr : &laddr);

	prot = (type == SOCK_DGRAM ? 0 : PF_UNSPEC);

	memset(saddr, 0, sizeof(struct sockaddr_in));

	saddr->sin_family = PF_INET;
	saddr->sin_addr.s_addr = htonl(INADDR_ANY);
	saddr->sin_port = htons(*port);

#ifdef TRASH
	if (broadcast && (net = getnetbyname("localnet")) == NULL) {
		perror("getnetbyname error");
		return(0);
	}
#else
	if (broadcast && !(hp = gethostbyname("eiche.informatik.hu-berlin.de"))) {
		perror("gethostbyname error");
		return (0);
	}
#endif

	if (broadcast)
#ifdef TRASH
		memcpy((char *) &saddr->sin_addr,
				(char *) inet_makeaddr(INADDR_ANY, net->n_net),
				sizeof(struct in_addr));
#else
		memcpy((char *) &saddr->sin_addr,
				(char *) inet_makeaddr(inet_netof(hp->h_addr), INADDR_ANY),
		sizeof(struct in_addr));
#endif

	if ((s = socket(PF_INET, type, prot)) < 0) {
		perror("socket error");
		return (0);
	}

	/*
	 * socket port doesn't linger after close (yes, both options needed)
	 */
#ifdef TRASH
	if (setsockopt (s, SOL_SOCKET, SO_REUSEADDR, (char *) &on, sizeof(on)) < 0) {
		perror ("setsockopt reuse error");
		close(s);
		return(0);
	}
#endif
	l.l_onoff = 0;
	if (setsockopt(s, SOL_SOCKET, DSM_LINGER, (char *) &l,
			sizeof(struct linger)) < 0) {
		perror("setsockopt linger error");
		close(s);
		return (0);
	}

	/*
	 * set size of socket buffers
	 */
	if (bufsize != 0
			&& setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char *) &bufsize,
					sizeof(bufsize)) < 0)
		perror("setsockopt RCVBUF error");
	if (bufsize != 0
			&& setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char *) &bufsize,
					sizeof(bufsize)) < 0)
		perror("setsockopt SNDBUF error");

	if (bind(s, (struct sockaddr *) saddr, sizeof(struct sockaddr))) {
#ifdef DEBUG
		perror("bind error");
#endif
		close(s);
		return (0);
	}

	if (*port == 0) {
		len = sizeof(name);
		if (getsockname(s, (struct sockaddr *) &name, &len) < 0) {
			perror("getsockname error");
			close(s);
			return (0);
		}
		*port = ntohs(name.sin_port);
	}

	if (type == SOCK_STREAM) {
		if (listen(s, 5)) {
			perror("listen error");
			close(s);
			return (0);
		}
	}

	return (s);
}

/*
 * dsm_server_accept - accept new connections on medium "(s, type, port)"
 * optionally for "broadcast", optionally fill in comm. address "gaddr"
 * and return length of acceptable msg in msglen (or 0 if not supported)
 * or return the peeked at (but not yet read) part of the message in
 * "*buf" with a maximum of "maxlen"; *buf == NULL indicates no peeking done
 * return a new handle "s" (used for read/write)
 */
int dsm_server_accept(int s, int type, int port, int broadcast,
		struct sockaddr_in *gaddr, int *msglen, char **buf, int maxlen) {
	int s2;
	struct hostent *hp;
	int prot, len, on = 1;
	struct linger l;
	struct sockaddr_in laddr, *saddr;

#ifdef PROF
	prof_checkpoint(">> sock_msg/server_accept");
#endif

	if (gaddr == NULL) {
		memset(&laddr, 0, sizeof(struct sockaddr_in));
		laddr.sin_family = PF_INET;
		laddr.sin_addr.s_addr = htonl(INADDR_ANY);
		laddr.sin_port = htons(port);
		saddr = &laddr;
	} else
		saddr = gaddr;

	prot = (type == SOCK_DGRAM ? 0 : PF_UNSPEC);

	*msglen = 0; /* determine length in 1st read, get rest in 2nd read */
	if (type == SOCK_DGRAM) { /* determine length here, then issue 2 reads */
		len = sizeof(struct sockaddr_in);
		while ((*msglen = recvfrom(s, *buf, maxlen, MSG_PEEK,
				(struct sockaddr *) saddr, &len)) == -1)
			if (errno != EINTR) {
				*msglen = 0;
				perror("recvfrom error");
				return (0);
			}
	} else if (type == SOCK_STREAM) {
		*buf = NULL;
		len = sizeof(struct sockaddr_in);
		while ((s2 = accept(s, (struct sockaddr *) saddr, &len)) < 0)
			if (errno != EINTR) {
#ifdef DEBUG
				perror("accept error");
#endif
				return (0);
			}
#ifdef PROF
		prof_checkpoint("<< sock_msg/server_accept");
#endif
		return (s2);
	}
#ifdef PROF
	prof_checkpoint("<< sock_msg/server_accept");
#endif
	return (s);
}

/*
 * dsm_client_open - open communiction medium "(type, port)" to host "dest"
 * or optionally for "broadcast", optionally fill in comm. address "gaddr"
 * if
 * return a handle "s" (used for read/write)
 */
int dsm_client_open(int type, int port, int broadcast, char *dest,
		struct sockaddr_in *gaddr) {
	struct hostent *hp;
	int s;
	int prot;
	char on;
	struct sockaddr_in laddr, *saddr;

#ifdef PROF
	prof_checkpoint(">> sock_msg/client_open");
#endif

	saddr = (gaddr != NULL ? gaddr : &laddr);

	prot = (type == SOCK_DGRAM ? 0 : PF_UNSPEC);

	memset(saddr, 0, sizeof(struct sockaddr_in));

	if (!(hp = gethostbyname(dest))) {
		perror("gethostbyname");
		return (0);
	}

	saddr->sin_addr = *(struct in_addr *) *hp->h_addr_list;

	saddr->sin_family = hp->h_addrtype;
	saddr->sin_port = htons(port);

	if ((s = socket(hp->h_addrtype, type, prot)) < 0) {
		perror("socket error");
		return (0);
	}

	/*
	 * socket port for broadcast
	 */
	on = 1;
	if (broadcast
			&& setsockopt(s, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on)) < 0) {
		perror("setsockopt broadcast error");
		return (0);
	}

	if (broadcast)
		saddr->sin_addr.s_addr = htonl(INADDR_ANY);

	if (broadcast
			&& bind(s, (struct sockaddr *) saddr, sizeof(struct sockaddr))) {
		perror("bind error");
		return (0);
	}

	if ((type == SOCK_STREAM/* || gaddr != NULL*/))
		while (connect(s, (struct sockaddr *) saddr, sizeof(struct sockaddr)))
			if (errno != EINTR) {
#ifndef DEBUG
				perror("connect error");
#endif
				dsm_close(s);
				return (0);
			}

#ifdef PROF
	prof_checkpoint("<< sock_msg/client_open");
#endif

	return (s);
}

/*
 * dsm_write - transmit msg "buf" with length "len" on medium "(s, type)"
 * optionally using comm. address "saddr"
 * return msg length or -1 upon error
 */
int dsm_write(int s, int type, char *buf, int len, struct sockaddr *saddr) {
	int ret;

#ifdef PROF
	prof_checkpoint(">> sock_msg/write");
#endif

	if (type == SOCK_DGRAM && saddr != NULL) {
		if ((ret = sendto(s, buf, len, 0, saddr, sizeof(struct sockaddr)))
				< 0) {
			perror("sendto error");
			return (-1);
		}
	} else if ((ret = write(s, buf, len)) < 0) {
		perror("write error");
		return (-1);
	}
#ifdef PROF
	prof_checkpoint("<< sock_msg/write");
#endif

	return (ret);
}

/*
 * dsm_read - receive msg "buf" of max. length "len" from medium "(s, type)"
 * optionally fill in comm. address "saddr"
 * return msg length or -1 upon error
 */
int dsm_read(int s, int type, char *buf, int len, struct sockaddr *saddr) {
	int slen = sizeof(struct sockaddr);

#ifdef PROF
	prof_checkpoint(">> sock_msg/read");
#endif

#ifdef DEBUG
	fprintf(stderr, "dsm_read1 %x\n", buf); fflush(stderr);
#endif
	if (type == SOCK_DGRAM && saddr != NULL) {
		if ((len = recvfrom(s, buf, len, 0, saddr, &slen)) < 0)
			perror("read error");
	} else if ((len = read(s, buf, len)) < 0)
		perror("read error");
#ifdef DEBUG
	fprintf(stderr, "dsm_read2 %d\n", errno); fflush(stderr);
#endif
#ifdef PROF
	prof_checkpoint("<< sock_msg/read");
#endif

	return (len);
}

/*
 * dsm_close - close communication medium "s"
 * used by: commserv.c (dsm_commexit), mem_comm.c (dsm_commexit)
 */
int dsm_close(int s) {
	int ret;

#ifdef PROF
	prof_checkpoint(">> sock_msg/close");
#endif

	if (ret = close(s))
		perror("close error");

#ifdef PROF
	prof_checkpoint("<< sock_msg/close");
#endif

	return (ret);
}

/*
 * dsm_client_close - close communication medium "s" to a client
 * used by: commserv.c (dsm_commSend), mad_msg.c (dsm_client_open),
 *          mem_comm.c (dsm_commSend), reliable.c (dsm_ReliableWriteMsg),
 *          sock_msg.c (dsm_client_open)
 */
int dsm_client_close(int s) {
	int ret;

#ifdef PROF
	prof_checkpoint(">> sock_msg/client_close");
#endif

	if (ret = close(s))
		perror("close error");

#ifdef PROF
	prof_checkpoint("<< sock_msg/client_close");
#endif

	return (ret);
}

/*
 * dsm_server_close - close communication medium "s" to a server
 * used by: commserv.c (dsm_commServer), mem_comm.c (dsm_commServer)
 */
int dsm_server_close(int s) {
	int ret;

#ifdef PROF
	prof_checkpoint(">> sock_msg/server_close");
#endif

	if (ret = close(s))
		perror("close error");

#ifdef PROF
	prof_checkpoint("<< sock_msg/server_close");
#endif

	return (ret);
}
