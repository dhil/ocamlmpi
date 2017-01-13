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

/* Utility functions on arrays */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include "camlmpi.h"

void caml_mpi_decode_intarray(value data, mlsize_t len)
{
  CAMLparam1(data);
  mlsize_t i;
  for (i = 0; i < len; i++) caml_modify_field(data, i, Long_field(data, i));
  CAMLreturn0;
}

void caml_mpi_encode_intarray(value data, mlsize_t len)
{
  CAMLparam1(data);
  CAMLlocal1(e);
  mlsize_t i;
  for (i = 0; i < len; i++) {
    caml_read_field(data, i, &e);
    caml_modify_field(data, i, Val_long(e));
  }
  CAMLreturn0;
}

#ifdef ARCH_ALIGN_DOUBLE

double * caml_mpi_input_floatarray(value data, mlsize_t len)
{
  CAMLparam1(data);
  double * d = stat_alloc(len * sizeof(double));
  bcopy((double *) data, d, len * sizeof(double));
  CAMLreturnT(double*, d);
}

double * caml_mpi_output_floatarray(value data, mlsize_t len)
{
  CAMLparam1(data);
  CAMLreturnT(double*, stat_alloc(len * sizeof(double)));
}

void caml_mpi_free_floatarray(double * d)
{
  CAMLparam0();
  if (d != NULL) caml_stat_free(d);
  CAMLreturn0;
}

void caml_mpi_commit_floatarray(double * d, value data, mlsize_t len)
{
  CAMLparam1(data);
  if (d != NULL) {
    bcopy(d, (double *) data, len * sizeof(double));
    caml_stat_free(d);
  }
  CAMLreturn0;
}

double * caml_mpi_input_floatarray_at_node(value data, mlsize_t len,
                                           value root, value comm)
{
  CAMLparam3(data, root, comm);
  int myrank;
  double *array = NULL;
  MPI_Comm_rank(Comm_val(comm), &myrank);
  if (myrank == Int_val(root))
    array = caml_mpi_input_floatarray(data, len);
  CAMLreturnT(double *, array);
}

double * caml_mpi_output_floatarray_at_node(value data, mlsize_t len,
                                           value root, value comm)
{
  CAMLparam3(data, root, comm);
  int myrank;
  double *array = NULL;
  MPI_Comm_rank(Comm_val(comm), &myrank);
  if (myrank == Int_val(root))
    array = caml_mpi_output_floatarray(data, len);
  CAMLreturnT(double *, array);
}

#endif
