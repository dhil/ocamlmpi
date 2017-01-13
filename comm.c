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

/* Handling of communicators */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include "camlmpi.h"

static void caml_mpi_finalize_comm(value v)
{
  CAMLparam1(v);
  MPI_Comm_free(&Comm_val(v));
  CAMLreturn0;
}

value caml_mpi_alloc_comm(MPI_Comm c)
{
  CAMLparam0();
  CAMLlocal1(res);
  res = alloc_final(1 + (sizeof(MPI_Comm) + sizeof(value) - 1) / sizeof(value),
                    caml_mpi_finalize_comm, 1, 100);
  Comm_val(res) = c;
  CAMLreturn(res);
}

value caml_mpi_get_comm_world(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(caml_mpi_alloc_comm(MPI_COMM_WORLD));
}

value caml_mpi_comm_size(value comm)
{
  CAMLparam1(comm);
  int size;
  MPI_Comm_size(Comm_val(comm), &size);
  CAMLreturn(Val_int(size));
}

value caml_mpi_comm_rank(value comm)
{
  CAMLparam1(comm);
  int rank;
  MPI_Comm_rank(Comm_val(comm), &rank);
  CAMLreturn(Val_int(rank));
}

value caml_mpi_comm_compare(value comm1, value comm2)
{
  CAMLparam2(comm1, comm2);
  int res;
  MPI_Comm_compare(Comm_val(comm1), Comm_val(comm2), &res);
  CAMLreturn(Val_bool(res));
}

value caml_mpi_comm_split(value comm, value color, value key)
{
  CAMLparam3(comm, color, key);
  MPI_Comm newcomm;
  MPI_Comm_split(Comm_val(comm), Int_val(color), Int_val(key), &newcomm);
  CAMLreturn(caml_mpi_alloc_comm(newcomm));
}

value caml_mpi_get_undefined(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(Val_int(MPI_UNDEFINED));
}

value caml_mpi_cart_create(value comm, value vdims, value vperiods,
                           value reorder)
{
  CAMLparam4(comm, vdims, vperiods, reorder);
  int ndims = Wosize_val(vdims);
  int * dims = caml_stat_alloc(ndims * sizeof(int));
  int * periods = caml_stat_alloc(ndims * sizeof(int));
  int i;
  MPI_Comm newcomm;

  for (i = 0; i < ndims; i++) dims[i] = Int_field(vdims, i);
  for (i = 0; i < ndims; i++) periods[i] = Int_field(vperiods, i);
  MPI_Cart_create(Comm_val(comm), ndims, dims, periods, 
                  Bool_val(reorder), &newcomm);
  caml_stat_free(dims);
  caml_stat_free(periods);
  CAMLreturn(caml_mpi_alloc_comm(newcomm));
}

value caml_mpi_dims_create(value vnnodes, value vdims)
{
  CAMLparam2(vnnodes, vdims);
  CAMLlocal1(res);
  int ndims = Wosize_val(vdims);
  int * dims = caml_stat_alloc(ndims * sizeof(int));
  int i;

  for (i = 0; i < ndims; i++) dims[i] = Int_field(vdims, i);
  MPI_Dims_create(Int_val(vnnodes), ndims, dims);
  res = alloc_tuple(ndims);
  for (i = 0; i < ndims; i++) caml_initialize_field(res, i, Val_int(dims[i]));

  caml_stat_free(dims);
  CAMLreturn(res);
}

value caml_mpi_cart_rank(value comm, value vcoords)
{
  CAMLparam2(comm, vcoords);
  int ndims = Wosize_val(vcoords);
  int *coords = caml_stat_alloc(ndims * sizeof(int));
  int i, rank;

  for (i = 0; i < ndims; i++) coords[i] = Int_field(vcoords, i);
  MPI_Cart_rank(Comm_val(comm), coords, &rank);
  caml_stat_free(coords);
  CAMLreturn(Val_int(rank));
}

value caml_mpi_cart_coords(value comm, value rank)
{
  CAMLparam2(comm, rank);
  CAMLlocal1(res);
  int ndims, i;
  int *coords;

  MPI_Cartdim_get(Comm_val(comm), &ndims);
  coords = caml_stat_alloc(ndims * sizeof(int));
  MPI_Cart_coords(Comm_val(comm), Int_val(rank), ndims, coords);
  res = alloc_tuple(ndims);
  for (i = 0; i < ndims; i++) caml_initialize_field(res, i, Val_int(coords[i]));
  caml_stat_free(coords);
  CAMLreturn(res);
}

value caml_mpi_comm_create(value comm, value group)
{
  CAMLparam2(comm, group);
  MPI_Comm newcomm;
  MPI_Comm_create(Comm_val(comm), Group_val(group), &newcomm);
  CAMLreturn(caml_mpi_alloc_comm(newcomm));
}

value caml_mpi_comm_abort(value comm, value exit_code) {
  CAMLparam2(comm, exit_code);
  int ecode;
  ecode = MPI_Abort(Comm_val(comm), Int_val(exit_code));
  CAMLreturn(Val_int(ecode));
}
