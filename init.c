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

/* Initialization and error handling */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/callback.h>
#include "camlmpi.h"

/* Error handling */

static caml_root caml_mpi_exn = NULL;

static void caml_mpi_error_handler(MPI_Comm * comm, int * errcode, ...)
{
  CAMLparam0();
  CAMLlocal1(msg);
  
  char errmsg[MPI_MAX_ERROR_STRING + 1];
  int resultlen;

  MPI_Error_string(*errcode, errmsg, &resultlen);
  msg = copy_string(errmsg);
  if (caml_mpi_exn == NULL) {
    caml_mpi_exn = caml_named_root("Mpi.Error");
    if (caml_mpi_exn == NULL)
      invalid_argument("Exception MPI.Error not initialized");
    else
      raise_with_arg(caml_read_root(caml_mpi_exn), msg);
  }

  CAMLreturn0;
}

/* Initialization and finalization */

value caml_mpi_init(value arguments)
{
  CAMLparam1(arguments);
  int argc, i;
  char ** argv;
  MPI_Errhandler hdlr;

  argc = Wosize_val(arguments);
  argv = caml_stat_alloc((argc + 1) * sizeof(char *));
  for (i = 0; i < argc; i++) argv[i] = String_val(Field_imm(arguments, i));
  argv[i] = NULL;
  MPI_Init(&argc, &argv);
  /* Register an error handler */
  //MPI_Errhandler_create((MPI_Handler_function *)caml_mpi_error_handler, &hdlr);
  //MPI_Errhandler_set(MPI_COMM_WORLD, hdlr);
  
  CAMLreturn(Val_unit);
}

value caml_mpi_finalize(value unit)
{
  CAMLparam1(unit);
  MPI_Finalize();
  CAMLreturn(Val_unit);
}

value caml_mpi_wtime(value unit)
{
  CAMLparam1(unit);
  CAMLreturn(copy_double(MPI_Wtime()));
}
