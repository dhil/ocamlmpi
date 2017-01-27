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

/* Group communication */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include "camlmpi.h"

/* Barrier synchronization */

value caml_mpi_barrier(value comm)
{
  MPI_Barrier(Comm_val(comm));
  return Val_unit;
}

/* Broadcast */

value caml_mpi_broadcast(value buffer, value root, value comm)
{
  CAMLparam3(buffer, root, comm);
  MPI_Bcast(Bytes_val(buffer), bytes_length(buffer), MPI_BYTE,
            Int_val(root), Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_broadcast_int(value data, value root, value comm)
{
  CAMLparam3(data, root, comm);
  long n = Long_val(data);
  MPI_Bcast(&n, 1, MPI_LONG, Int_val(root), Comm_val(comm));
  CAMLreturn(Val_long(n));
}

value caml_mpi_broadcast_float(value data, value root, value comm)
{
  CAMLparam3(data, root, comm);
  double d = Double_val(data);
  MPI_Bcast(&d, 1, MPI_DOUBLE, Int_val(root), Comm_val(comm));
  CAMLreturn(copy_double(d));
}

value caml_mpi_broadcast_intarray(value data, value root, value comm)
{
  CAMLparam3(data, root, comm);
  CAMLlocal1(buffer);
  MPI_Bcast((long*)Op_val(data), Wosize_val(data), MPI_LONG,
            Int_val(root), Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_broadcast_floatarray(value data, value root, value comm)
{
  CAMLparam3(data, root, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);
  MPI_Bcast(d, len, MPI_DOUBLE, Int_val(root), Comm_val(comm));
  caml_mpi_commit_floatarray(d, data, len);
  CAMLreturn(Val_unit);
}

/* Scatter */

static void caml_mpi_counts_displs(value lengths,
                                   /* out */ int ** counts,
                                   /* out */ int ** displs)
{
  CAMLparam1(lengths);
  int size, disp, i;

  size = Wosize_val(lengths);
  if (size > 0) {
    *counts = caml_stat_alloc(size * sizeof(int));
    *displs = caml_stat_alloc(size * sizeof(int));
    for (i = 0, disp = 0; i < size; i++) {
      (*counts)[i] = Int_field(lengths, i);
      (*displs)[i] = disp;
      disp += (*counts)[i];
    }
  } else {
    *counts = NULL;
    *displs = NULL;
  }
  CAMLreturn0;
}

value caml_mpi_scatter(value sendbuf, value sendlengths, 
                       value recvbuf,
                       value root, value comm)
{
  CAMLparam5(sendbuf, sendlengths, recvbuf, root, comm);
  int * sendcounts, * displs;

  caml_mpi_counts_displs(sendlengths, &sendcounts, &displs);
  MPI_Scatterv(Bytes_val(sendbuf), sendcounts, displs, MPI_BYTE,
               Bytes_val(recvbuf), bytes_length(recvbuf), MPI_BYTE,
               Int_val(root), Comm_val(comm));
  if (sendcounts != NULL) {
    caml_stat_free(sendcounts);
    caml_stat_free(displs);
  }
  CAMLreturn(Val_unit);
}

value caml_mpi_scatter_int(value data, value root, value comm)
{
  CAMLparam3(data, root, comm);
  CAMLlocal2(n, srcbuf);
  caml_read_field(data, 0, &srcbuf);
  
  MPI_Scatter((long*)Op_val(data), 1, MPI_LONG, /* (long*)Op_val(data) is a nasty hack -- it exploits that Stock OCaml and Multicore OCaml use the same layout for an array of integers */
              &n, 1, MPI_LONG,
              Int_val(root), Comm_val(comm));
  CAMLreturn(n);
}

value caml_mpi_scatter_float(value data, value root, value comm)
{
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * src = caml_mpi_input_floatarray(data, len);
  double dst;
  MPI_Scatter(src, 1, MPI_DOUBLE, &dst, 1, MPI_DOUBLE,
              Int_val(root), Comm_val(comm));
  caml_mpi_free_floatarray(src);
  return copy_double(dst);
}

value caml_mpi_scatter_intarray(value source, value dest,
                                value root, value comm)
{
  CAMLparam4(source, dest, root, comm);
  CAMLlocal2(srcbuf, destbuf);
  caml_read_field(source, 0, &srcbuf);

  mlsize_t len = Wosize_val(dest);
  
  MPI_Scatter(&srcbuf, len, MPI_LONG,
              &destbuf, len, MPI_LONG,
              Int_val(root), Comm_val(comm));
  
  caml_modify_field(dest, 0, destbuf);
  CAMLreturn(Val_unit);
}

value caml_mpi_scatter_floatarray(value source, value dest,
                                  value root, value comm)
{
  CAMLparam4(source, dest, root, comm);
  mlsize_t srclen = Wosize_val(source) / Double_wosize;
  mlsize_t len = Wosize_val(dest) / Double_wosize;
  double * src = caml_mpi_input_floatarray_at_node(source, srclen, root, comm);
  double * dst = caml_mpi_output_floatarray(dest, len);

  MPI_Scatter(src, len, MPI_DOUBLE, dst, len, MPI_DOUBLE,
              Int_val(root), Comm_val(comm));
  caml_mpi_free_floatarray(src);
  caml_mpi_commit_floatarray(dst, dest, len);
  CAMLreturn(Val_unit);
}

/* Gather */

void print_int_array(int *array, int length) {
  int i;
  for (i = 0; i < length; i++) printf("%d ", array[i]);
  printf("\n");
}

value caml_mpi_gather(value sendbuf,
                      value recvbuf, value recvlengths,
                      value root, value comm)
{
  CAMLparam5(sendbuf, recvbuf, recvlengths, root, comm);
  int *recvcounts, *displs;  

  caml_mpi_counts_displs(recvlengths, &recvcounts, &displs);
  MPI_Gatherv(Bytes_val(sendbuf), bytes_length(sendbuf), MPI_BYTE,
              Bytes_val(recvbuf), recvcounts, displs, MPI_BYTE,
              Int_val(root), Comm_val(comm));
  if (recvcounts != NULL) {
    caml_stat_free(recvcounts);
    caml_stat_free(displs);
  }
  CAMLreturn(Val_unit);
}

value caml_mpi_gather_int(value data, value result, value root, value comm)
{
  CAMLparam4(data, result, root, comm);
  MPI_Gather(&data, 1, MPI_LONG,
             (long*)Op_val(result), 1, MPI_LONG,
             Int_val(root), Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_gather_intarray(value data, value result,
                               value root, value comm)
{
  CAMLparam4(data, result, root, comm);
  CAMLlocal2(srcbuf, destbuf);
  caml_read_field(data, 0, &srcbuf);

  mlsize_t len = Wosize_val(data);
  
  MPI_Gather(&srcbuf, len, MPI_LONG,
             &destbuf, len, MPI_LONG,
             Int_val(root), Comm_val(comm));

  caml_modify_field(result, 0, destbuf);
  CAMLreturn(Val_unit);
}

value caml_mpi_gather_float(value data, value result, value root, value comm)
{
  CAMLparam4(data, result, root, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  mlsize_t reslen = Wosize_val(result) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);
  double * res =
    caml_mpi_output_floatarray_at_node(result, reslen, root, comm);
  MPI_Gather(d, len, MPI_DOUBLE, res, len, MPI_DOUBLE,
             Int_val(root), Comm_val(comm));
  caml_mpi_free_floatarray(d);
  caml_mpi_commit_floatarray(res, result, reslen);
  CAMLreturn(Val_unit);
}

/* Gather to all */

value caml_mpi_allgather(value sendbuf,
                         value recvbuf, value recvlengths,
                         value comm)
{
  CAMLparam4(sendbuf, recvbuf, recvlengths, comm);
  int * recvcounts, * displs;

  caml_mpi_counts_displs(recvlengths, &recvcounts, &displs);
  MPI_Allgatherv(Bytes_val(sendbuf), bytes_length(sendbuf), MPI_BYTE,
                 Bytes_val(recvbuf), recvcounts, displs, MPI_BYTE,
                 Comm_val(comm));
  caml_stat_free(recvcounts);
  caml_stat_free(displs);
  
  CAMLreturn(Val_unit);
}

value caml_mpi_allgather_int(value data, value result, value comm)
{
  CAMLparam3(data, result, comm);
  CAMLlocal1(buffer);
  caml_read_field(data, 0, &buffer);
  
  MPI_Allgather(&data, 1, MPI_LONG,
                &buffer, 1, MPI_LONG,
                Comm_val(comm));
  CAMLreturn(Val_unit);
}

value caml_mpi_allgather_intarray(value data, value result, value comm)
{
  CAMLparam3(data, result, comm);
  CAMLlocal2(srcbuf, destbuf);
  caml_read_field(data, 0, &srcbuf);

  mlsize_t len = Wosize_val(data);
  
  MPI_Allgather(&srcbuf, len, MPI_LONG,
                &destbuf, len, MPI_LONG,
                Comm_val(comm));

  caml_modify_field(result, 0, destbuf);
  CAMLreturn(Val_unit);
}

value caml_mpi_allgather_float(value data, value result, value comm)
{
  CAMLparam3(data, result, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  mlsize_t reslen = Wosize_val(result) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);
  double * res = caml_mpi_output_floatarray(result, reslen);

  MPI_Allgather(d, len, MPI_DOUBLE, res, len, MPI_DOUBLE,
                Comm_val(comm));
  caml_mpi_free_floatarray(d);
  caml_mpi_commit_floatarray(res, result, reslen);
  CAMLreturn(Val_unit);
}

/* Reduce */

static MPI_Op reduce_intop[] =
  { MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD, MPI_BAND, MPI_BOR, MPI_BXOR };
static MPI_Op reduce_floatop[] =
  { MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD };

value caml_mpi_reduce_int(value data, value op, value root, value comm)
{
  CAMLparam4(data, op, root, comm);
  long d = Long_val(data);
  long r = 0;
  MPI_Reduce(&d, &r, 1, MPI_LONG,
             reduce_intop[Int_val(op)], Int_val(root), Comm_val(comm));
  CAMLreturn(Val_long(r));
}

value caml_mpi_reduce_intarray(value data, value result, value op,
                               value root, value comm)
{
  CAMLparam5(data, result, op, root, comm);
  CAMLlocal2(srcbuf, destbuf);
  mlsize_t len = Wosize_val(data);
  mlsize_t reslen = Wosize_val(result);
  int i, myrank;
  /* Decode data at all nodes in place */
  caml_mpi_decode_intarray(data, len);
  /* Do the reduce */
  caml_read_field(data, 0, &srcbuf);
  MPI_Reduce(&srcbuf, &destbuf, len, MPI_LONG,
             reduce_intop[Int_val(op)], Int_val(root), Comm_val(comm));
  caml_modify_field(result, 0, destbuf);
  /* Re-encode data at all nodes in place */
  caml_mpi_encode_intarray(data, len);
  /* At root node, also encode result */
  MPI_Comm_rank(Comm_val(comm), &myrank);
  if (myrank == Int_val(root)) caml_mpi_encode_intarray(result, reslen);
  CAMLreturn(Val_unit);
}

value caml_mpi_reduce_float(value data, value op, value root, value comm)
{
  CAMLparam4(data, op, root, comm);
  double d = Double_val(data);
  double r = 0.0;
  MPI_Reduce(&d, &r, 1, MPI_DOUBLE,
             reduce_floatop[Int_val(op)], Int_val(root), Comm_val(comm));
  CAMLreturn(copy_double(r));
}

value caml_mpi_reduce_floatarray(value data, value result, value op,
                            value root, value comm)
{
  CAMLparam5(data, result, op, root, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);
  double * res = caml_mpi_output_floatarray(result, len);

  MPI_Reduce(d, res, len, MPI_DOUBLE,
             reduce_floatop[Int_val(op)], Int_val(root), Comm_val(comm));
  caml_mpi_free_floatarray(d);
  caml_mpi_commit_floatarray(res, result, len);
  CAMLreturn(Val_unit);
}

/* Allreduce */

value caml_mpi_allreduce_int(value data, value op, value comm)
{
  CAMLparam3(data, op, comm);
  long d = Long_val(data);
  long r;
  MPI_Allreduce(&d, &r, 1, MPI_LONG,
                reduce_intop[Int_val(op)], Comm_val(comm));
  CAMLreturn(Val_long(r));
}

value caml_mpi_allreduce_intarray(value data, value result, value op,
                                  value comm)
{
  CAMLparam4(data, result, op, comm);
  CAMLlocal2(srcbuf, destbuf);
  mlsize_t len = Wosize_val(data);
  mlsize_t reslen = Wosize_val(result);
  /* Decode data at all nodes in place */
  caml_mpi_decode_intarray(data, len);
  /* Do the reduce */
  caml_read_field(data, 0, &srcbuf);
  MPI_Allreduce(&srcbuf, &destbuf, len, MPI_LONG,
                reduce_intop[Int_val(op)], Comm_val(comm));
  caml_modify_field(result, 0, destbuf);
  /* Re-encode data at all nodes in place */
  caml_mpi_encode_intarray(data, len);
  /* Re-encode result at all nodes in place */
  caml_mpi_encode_intarray(result, reslen);
  CAMLreturn(Val_unit);
}

value caml_mpi_allreduce_float(value data, value op, value comm)
{
  CAMLparam3(data, op, comm);
  double d = Double_val(data);
  double r;
  MPI_Allreduce(&d, &r, 1, MPI_DOUBLE,
                reduce_floatop[Int_val(op)], Comm_val(comm));
  CAMLreturn(copy_double(r));
}

value caml_mpi_allreduce_floatarray(value data, value result, value op,
                                    value comm)
{
  CAMLparam4(data, result, op, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);

  mlsize_t reslen = Wosize_val(result) / Double_wosize;
  double * res = caml_mpi_output_floatarray(result, reslen);

  MPI_Allreduce(d, res, len, MPI_DOUBLE,
                reduce_floatop[Int_val(op)], Comm_val(comm));
  caml_mpi_free_floatarray(d);
  caml_mpi_commit_floatarray(res, result, reslen);
  CAMLreturn(Val_unit);
}

/* Scan */

value caml_mpi_scan_int(value data, value op, value comm)
{
  CAMLparam3(data, op, comm);
  long d = Long_val(data);
  long r;

  MPI_Scan(&d, &r, 1, MPI_LONG, reduce_intop[Int_val(op)], Comm_val(comm));
  CAMLreturn(Val_long(r));
}

value caml_mpi_scan_intarray(value data, value result, value op, value comm)
{
  CAMLparam4(data, result, op, comm);
  CAMLlocal2(srcbuf, destbuf);
  mlsize_t len = Wosize_val(data);

  /* Decode data at all nodes in place */
  caml_mpi_decode_intarray(data, len);
  /* Do the scan */
  caml_read_field(data, 0, &srcbuf);
  MPI_Scan(&srcbuf, &destbuf, len, MPI_LONG,
           reduce_intop[Int_val(op)], Comm_val(comm));
  caml_modify_field(result, 0, destbuf);
  /* Re-encode data at all nodes in place */
  caml_mpi_encode_intarray(data, len);
  /* Encode result */
  caml_mpi_encode_intarray(result, len);
  CAMLreturn(Val_unit);
}

value caml_mpi_scan_float(value data, value op, value comm)
{
  CAMLparam3(data, op, comm);
  double d = Double_val(data), r;

  MPI_Scan(&d, &r, 1, MPI_DOUBLE,
           reduce_floatop[Int_val(op)], Comm_val(comm));
  CAMLreturn(copy_double(r));
}

value caml_mpi_scan_floatarray(value data, value result, value op, value comm)
{
  CAMLparam4(data, result, op, comm);
  mlsize_t len = Wosize_val(data) / Double_wosize;
  double * d = caml_mpi_input_floatarray(data, len);
  double * res = caml_mpi_output_floatarray(result, len);

  MPI_Scan(d, res, len, MPI_DOUBLE,
           reduce_floatop[Int_val(op)], Comm_val(comm));
  caml_mpi_free_floatarray(d);
  caml_mpi_commit_floatarray(res, result, len);
  CAMLreturn(Val_unit);
}

