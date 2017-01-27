/***********************************************************************/
/*                                                                     */
/*                         The Caml/MPI interface                      */
/*                                                                     */
/*            Xavier Leroy, projet Cristal, INRIA Rocquencourt         */
/*                                                                     */
/*  Copyright 1998 Institut National de Recherche en Informatique et   */
/*  en Automatique.  All rights reserved.  This file is distributed    */
/*  under the terms of the GNU Library General Public License, with    */
/*  the special exception on linking described in file LICENSE.        */
/*                                                                     */
/***********************************************************************/

/* $Id$ */

/* Point-to-point communication */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/signals.h>
#include <stdio.h>
#include "camlmpi.h"

extern void output_value_to_malloc(value v, value flags,
                                   /*out*/ char ** buf, 
                                   /*out*/ long * len);
extern value input_value_from_malloc(char * data, long ofs);

#define Val_none Val_int(0)

/*#define Some_val(v) Field(v,0)*/

static inline value Val_some( value v )
{
  CAMLparam1( v );
  CAMLlocal1( some );
  some = caml_alloc(1, 0);
  Store_field( some, 0, v );
  CAMLreturn( some );
}

/* Sending */

value caml_mpi_send(value data, value flags,
                    value dest, value tag, value vcomm)
{
  CAMLparam5(data, flags, dest, tag, vcomm);
  //MPI_Comm comm = Comm_val(vcomm);
  char * buffer;
  long len;

  Begin_root(vcomm)             /* prevent deallocation of communicator */
    output_value_to_malloc(data, flags, &buffer, &len);
    /* This also allocates the buffer */
    enter_blocking_section();
    MPI_Send(buffer, len, MPI_BYTE, Int_val(dest), Int_val(tag), Comm_val(vcomm));
    leave_blocking_section();
  End_roots();
  free(buffer);
  CAMLreturn(Val_unit);
}


value caml_mpi_send_int(value data, value dest, value tag, value comm)
{
  CAMLparam4(data, dest, tag, comm);
  long n = Long_val(data);
  
  MPI_Send(&n, 1, MPI_LONG, Int_val(dest), Int_val(tag), Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_send_intarray(value data, value dest, value tag, value comm)
{
  CAMLparam4(data, dest, tag, comm);
  CAMLlocal1(buffer);
  caml_read_field(data, 0, &buffer);
  
  MPI_Send(&buffer, Wosize_val(data), MPI_LONG,
           Int_val(dest), Int_val(tag), Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_send_float(value data, value dest, value tag, value comm)
{
  CAMLparam4(data, dest, tag, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);

  MPI_Send(d, len, MPI_DOUBLE, Int_val(dest), Int_val(tag), Comm_val(comm));
  caml_mpi_free_floatarray(d);
  CAMLreturn(Val_unit);
}

/* Probe for pending messages and determine length */

value caml_mpi_probe(value source, value tag, value comm)
{
  CAMLparam3(source, tag, comm);
  CAMLlocal1(res);
  MPI_Status status;
  int count;

  MPI_Probe(Int_val(source), Int_val(tag), Comm_val(comm), &status);
  MPI_Get_count(&status, MPI_BYTE, &count);
  res = alloc_tuple(3);
  caml_initialize_field(res, 0, Val_int(count));
  caml_initialize_field(res, 1, Val_int(status.MPI_SOURCE));
  caml_initialize_field(res, 2, Val_int(status.MPI_TAG));

  CAMLreturn(res);
}

value caml_mpi_iprobe(value source, value tag, value comm)
{
  CAMLparam3(source, tag, comm);
  CAMLlocal1(res);
  MPI_Status status;
  int count, flag;

  MPI_Iprobe(Int_val(source), Int_val(tag), Comm_val(comm), &flag, &status);

  if (flag) {
    MPI_Get_count(&status, MPI_BYTE, &count);
    res = alloc_tuple(3);
    caml_initialize_field(res, 0, Val_int(count));
    caml_initialize_field(res, 1, Val_int(status.MPI_SOURCE));
    caml_initialize_field(res, 2, Val_int(status.MPI_TAG));
  }
  
  CAMLreturn(flag ? Val_some(res) : Val_none);
}

/* Receive */

value caml_mpi_receive(value vlen, value source, value tag, value vcomm)
{
  CAMLparam4(vlen, source, tag, vcomm);
  CAMLlocal1(res);
  //MPI_Comm comm = Comm_val(vcomm);
  mlsize_t len = Long_val(vlen);
  char * buffer;
  MPI_Status status;

  Begin_root(vcomm)             /* prevent deallocation of communicator */
    buffer = caml_stat_alloc(len);
    enter_blocking_section();
    MPI_Recv(buffer, len, MPI_BYTE,
             Int_val(source), Int_val(tag), Comm_val(vcomm), &status);
    leave_blocking_section();
    res = input_value_from_malloc(buffer, 0);
    /* This also deallocates the buffer */
  End_roots();
  CAMLreturn(res);
}

value caml_mpi_receive_int(value source, value tag, value comm)
{
  CAMLparam3(source, tag, comm);
  MPI_Status status;
  long n;

  MPI_Recv(&n, 1, MPI_LONG,
           Int_val(source), Int_val(tag), Comm_val(comm), &status);
  CAMLreturn(Val_long(n));
}

value caml_mpi_receive_intarray(value data, value source, value tag, value comm)
{
  CAMLparam4(data, source, tag, comm);
  CAMLlocal1(buffer);  
  caml_read_field(data, 0, &buffer);

  MPI_Status status;
  
  MPI_Recv(&buffer, Wosize_val(data), MPI_LONG,
           Int_val(source), Int_val(tag), Comm_val(comm), &status);
  CAMLreturn(Val_unit);
}

value caml_mpi_receive_float(value source, value tag, value comm)
{
  CAMLparam3(source, tag, comm);
  MPI_Status status;
  double d;

  MPI_Recv(&d, 1, MPI_DOUBLE,
           Int_val(source), Int_val(tag), Comm_val(comm), &status);
  CAMLreturn(copy_double(d));
}

value caml_mpi_receive_floatarray(value data, value source, value tag, value comm)
{
  CAMLparam4(data, source, tag, comm);
  MPI_Status status;
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_output_floatarray(data, len);

  MPI_Recv(d, len, MPI_DOUBLE,
           Int_val(source), Int_val(tag), Comm_val(comm), &status);
  caml_mpi_commit_floatarray(d, data, len);
  CAMLreturn(Val_unit);
}

/* Auxiliaries */

value caml_mpi_get_any_tag(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(Val_int(MPI_ANY_TAG));
}

value caml_mpi_get_any_source(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(Val_int(MPI_ANY_SOURCE));
}

/* Non-blocking comms */

static void caml_mpi_finalize_request(value v)
{
  CAMLparam1(v);
  /*printf("finalize req..");*/
  if (Request_req_val(v)!=MPI_REQUEST_NULL) {
    if (MPI_Request_free(&Request_req_val(v))!=MPI_SUCCESS)
      printf("ERROR: request cannot be freed!");
  }
  /*else
    printf("null request isn't freed\n");*/
  if (Buffer_req_val(v))
    caml_stat_free(Buffer_req_val(v)); /* free buffer */
  /*printf("done");*/
  CAMLreturn0;
}

value caml_mpi_alloc_request() 
{
  CAMLparam0();
  CAMLlocal1(res);
  /*printf("alloc req..");*/
  res = alloc_final(3, caml_mpi_finalize_request, 1, 100);
  Request_req_val(res) = MPI_REQUEST_NULL;
  Buffer_req_val(res) = 0;
  /*printf("done\n");*/ 
  CAMLreturn(res);
}

/*
static void caml_mpi_status(value v)
{
  MPI_Status_free(&Comm_val(v));
}

value caml_mpi_alloc_status(MPI_Request r)
{
  value res =
    alloc_final(1 + (sizeof(MPI_est) + sizeof(value) - 1) / sizeof(value),
                caml_mpi_finalize_request, 1, 100);
  Request_val(res) = r;
  return res;
}
*/

value caml_mpi_isend(value data, value flags,
                     value dest, value tag, value vcomm)
{
  CAMLparam5(data, flags, dest, tag, vcomm);
  CAMLlocal1(req);
  //MPI_Comm comm = Comm_val(vcomm);
  char *buffer;
  long len;
  req = caml_mpi_alloc_request();
  
  output_value_to_malloc(data, flags, &buffer, &len); //encode&alloc buffer
  enter_blocking_section();
  MPI_Isend(buffer, len, MPI_BYTE, Int_val(dest), Int_val(tag), Comm_val(vcomm), 
            &Request_req_val(req));
  leave_blocking_section();
  Buffer_req_val(req) = buffer; // store send buffer address 
  CAMLreturn(req);
}

value caml_mpi_isend_varlength(value data, value flags,
                               value dest, value tag, value vcomm)
{
  CAMLparam5(data, flags, dest, tag, vcomm);
  CAMLlocal3(result, lenreq, datareq);
  char *buffer;
  long len;
  long *lenbuf;
 
  MPI_Comm comm = Comm_val(vcomm);
  result = caml_alloc_tuple(2);
  lenreq = caml_mpi_alloc_request();
  datareq = caml_mpi_alloc_request();
  caml_initialize_field(result, 0, lenreq);
  caml_initialize_field(result, 1, datareq);
  output_value_to_malloc(data, flags, &buffer, &len); //encode&alloc buffer
  lenbuf = malloc(sizeof(long));
  *lenbuf = len;
  Buffer_req_val(lenreq) = (char*)lenbuf;
  Buffer_req_val(datareq) = buffer; // store send buffer address
  enter_blocking_section();
  MPI_Isend(Buffer_req_val(lenreq), 1, MPI_INT,
       	    Int_val(dest), Int_val(tag), comm, &Request_req_val(lenreq));
  MPI_Isend(buffer, len, MPI_BYTE, Int_val(dest), Int_val(tag), comm, 
            &Request_req_val(datareq));
  leave_blocking_section(); 
  
  CAMLreturn(result);
}


value caml_mpi_ireceive(value vlen, value src, value tag, value vcomm)
{
  CAMLparam4(vlen,src,tag,vcomm);
  CAMLlocal1(datareq);
  char *buffer;
  long len = Int_val(vlen);

  //MPI_Comm comm = Comm_val(vcomm);
  datareq = caml_mpi_alloc_request();
  Buffer_req_val(datareq) = buffer = malloc(len);
  enter_blocking_section();
  MPI_Irecv(buffer, len, MPI_BYTE, Int_val(src), Int_val(tag), Comm_val(vcomm), 
            &Request_req_val(datareq));
  leave_blocking_section(); 

  CAMLreturn(datareq);
}

value caml_mpi_ireceive_varlength(value src, value tag, value vcomm)
{
  CAMLparam3(src,tag,vcomm);
  CAMLlocal1(datareq);
  char *buffer;
  int len;
  MPI_Status status;

  MPI_Comm comm = Comm_val(vcomm);
  datareq = caml_mpi_alloc_request();
  enter_blocking_section();
  MPI_Recv(&len, 1, MPI_INT, Int_val(src), Int_val(tag), comm, &status);
  leave_blocking_section(); 
  Buffer_req_val(datareq) = buffer = malloc(len);
  enter_blocking_section();
  MPI_Irecv(buffer, len, MPI_BYTE, Int_val(src), Int_val(tag), comm, 
            &Request_req_val(datareq));
  leave_blocking_section(); 
 
  CAMLreturn(datareq);
}


value caml_mpi_wait(value req)
{
  CAMLparam1(req);
  int ret;

  enter_blocking_section();
  MPI_Status status;
  ret = MPI_Wait(&Request_req_val(req), &status);
  if (ret != MPI_SUCCESS)
    printf("ERROR: wait error!\n");
  leave_blocking_section();
  CAMLreturn(Val_unit);
}

value caml_mpi_wait_receive(value req)
{
  CAMLparam1(req);  
  CAMLlocal1(result);
  int ret;

  enter_blocking_section();
  MPI_Status status;
  ret = MPI_Wait(&Request_req_val(req), &status);
  if (ret != MPI_SUCCESS)
    printf("ERROR: wait error!\n");
  leave_blocking_section();
  result = input_value_from_malloc(Buffer_req_val(req), 0);
  Buffer_req_val(req) = 0; /* above deallocates buffer */
  CAMLreturn(result);
}
