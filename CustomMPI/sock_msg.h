/*
 * sock_msg.h
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
 *   $Id: sock_msg.h,v 1.13 2001/09/27 17:15:09 mueller Exp $
 *
 * DSM communication interface
 */

#include <netdb.h>
#include <arpa/nameser.h>
#include <resolv.h>

#ifndef _dsm_sock_msg
#define _dsm_sock_msg

#ifdef MAD
#define MADELEINE -1
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

/*
 * Communication types currently supported:
 * SOCK_STREAM:
 * . bidirectional read/writes, ordered, reliable
 * . dsm_close required for both handles (for accept and for read/write,
 *   retruned by server_open and server_accept, respectively.
 * . dsm_close required for client handle
 * SOCK_DGRAM:
 * . unidirectional (server reads, client writes), not ordered, unreliable
 * . dsm_close required for one handles (accept == read/write handle)
 * . dsm_close required for client handle
 * MADELEINE:
 * . bidirectional read/writes, ordered, reliable
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

/*
 * dsm_server_open - open a communication medium of "type",
 * either use fixed "port" or fill in "port" (if initially 0),
 * optionally for "broadcast" (if flag TRUE),
 * optionally fill in a medium address "gaddr" (if gaddr != NULL)
 * optionally set send and receive buffer size to bufsize (unless 0)
 * . with MADELEINE this is already done by mad_init
 * return a handle "s" (used for accept)
 * used by: comm.c (comm. init, dsm_commsend)
 */
int dsm_server_open(int type, int *port, int broadcast,
		struct sockaddr_in *gaddr, int bufsize);

/*
 * dsm_server_accept - accept new connections on medium "(s, type, port)"
 * optionally for "broadcast", optionally fill in comm. address "gaddr"
 * and return length of acceptable msg in msglen (or 0 if not supported)
 * or return the peeked at (but not yet read) part of the message in
 * "*buf" with a maximum of "maxlen"; *buf == NULL indicates no peeking done
 * . with MADELEINE it calls mad_receive only
 * return a new handle "s" (used for read/write)
 * used by: comm.c (comm. server, dsm_commsend)
 */
int dsm_server_accept(int s, int type, int port, int broadcast,
		struct sockaddr_in *gaddr, int *msglen, char **buf, int maxlen);

/*
 * dsm_client_open - open communiction medium "(type, port)" to host "dest"
 * or optionally for "broadcast", optionally fill in comm. address "gaddr"
 * . with MADELEINE it initializes the sending buffer(s)
 * return a handle "s" (used for read/write)
 * used by: comm.c (comm. server, dsm_commsend)
 */
int dsm_client_open(int type, int port, int broadcast, char *dest,
		struct sockaddr_in *gaddr);

/*
 * dsm_write - transmit msg "buf" with length "len" on medium "(s, type)"
 * optionally using comm. address "saddr"
 * return msg length or -1 upon error
 * used by: comm.c (comm. server, dsm_commsend)
 */
int dsm_write(int s, int type, char *buf, int len, struct sockaddr *saddr);

/*
 * dsm_read - receive msg "buf" of max. length "len" from medium "(s, type)"
 * optionally fill in comm. address "saddr"
 * return msg length or -1 upon error
 * used by: comm.c (comm. server, dsm_commsend)
 */
int dsm_read(int s, int type, char *buf, int len, struct sockaddr *saddr);
/*
 * dsm_write_delayed - transmit msg "buf" with length "len" to destination "dest"
 * using mode "mode"
 * return msg length or -1 upon error
 * used by: commserv.c (comm. server, dsm_commsend)
 */
int dsm_write_delayed(int s, int type, char *buf, int len,
		struct sockaddr *saddr, int dest, int mode);

/*
 * dsm_read_delayed - receive msg "buf" of max. length "len" in mode "mode"
 * return msg length or -1 upon error
 * used by: commserv.c (comm. server, dsm_commsend)
 */
int dsm_read_delayed(int s, int type, char *buf, int len,
		struct sockaddr *saddr, int mode);

/*
 * dsm_close - close communication medium "s"
 * used by: commserv.c (dsm_commexit), mem_comm.c (dsm_commexit)
 */
int dsm_close(int s);

/*
 * dsm_client_close - close communication medium "s" to a client
 * used by: commserv.c (dsm_commSend), mad_msg.c (dsm_client_open),
 *          mem_comm.c (dsm_commSend), reliable.c (dsm_ReliableWriteMsg),
 *          sock_msg.c (dsm_client_open)
 */
int dsm_client_close(int s);

/*
 * dsm_server_close - close communication medium "s" to a server
 * used by: commserv.c (dsm_commServer), mem_comm.c (dsm_commServer)
 */
int dsm_server_close(int s);

