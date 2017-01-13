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

/* Handling of groups */

#include <mpi.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include "camlmpi.h"

static void caml_mpi_finalize_group(value v)
{
  CAMLparam1(v);
  MPI_Group_free(&Group_val(v));
  CAMLreturn0;
}

value caml_mpi_alloc_group(MPI_Group g)
{
  CAMLparam0();
  CAMLlocal1(res);
  res = alloc_final(1 + (sizeof(MPI_Group) + sizeof(value) - 1) / sizeof(value),
                    caml_mpi_finalize_group, 1, 100);
  Group_val(res) = g;
  CAMLreturn(res);
}

value caml_mpi_group_size(value group)
{
  CAMLparam1(group);
  int size;
  MPI_Group_size(Group_val(group), &size);
  CAMLreturn(Val_int(size));
}

value caml_mpi_group_rank(value group)
{
  CAMLparam1(group);
  int size;
  MPI_Group_rank(Group_val(group), &size);
  CAMLreturn(Val_int(size));
}

value caml_mpi_group_translate_ranks(value group1, value ranks, value group2)
{
  CAMLparam3(group1, ranks, group2);
  CAMLlocal2(res, rank);
  int n = Wosize_val(ranks);
  int *ranks1 = caml_stat_alloc(n * sizeof(int));
  int *ranks2 = caml_stat_alloc(n * sizeof(int));
  int i;

  for (i = 0; i < n; i++) {
    caml_read_field(ranks, i, &rank);
    ranks1[i] = Int_val(rank);
  }
  MPI_Group_translate_ranks(Group_val(group1), n, ranks1,
                            Group_val(group2), ranks2);
  res = alloc(n, 0);
  for (i = 0; i < n; i++) caml_modify_field(res, i, Val_int(ranks2[i]));

  caml_stat_free(ranks1);
  caml_stat_free(ranks2);
  CAMLreturn(res);
}

value caml_mpi_comm_group(value comm)
{
  CAMLparam1(comm);
  MPI_Group group;
  MPI_Comm_group(Comm_val(comm), &group);
  CAMLreturn(caml_mpi_alloc_group(group));
}

value caml_mpi_group_union(value group1, value group2)
{
  CAMLparam2(group1, group2);
  MPI_Group group;
  MPI_Group_union(Group_val(group1), Group_val(group2), &group);
  CAMLreturn(caml_mpi_alloc_group(group));
}

value caml_mpi_group_difference(value group1, value group2)
{
  CAMLparam2(group1, group2);
  MPI_Group group;
  MPI_Group_difference(Group_val(group1), Group_val(group2), &group);
  CAMLreturn(caml_mpi_alloc_group(group));
}

value caml_mpi_group_intersection(value group1, value group2)
{
  CAMLparam2(group1, group2);
  MPI_Group group;
  MPI_Group_intersection(Group_val(group1), Group_val(group2), &group);
  CAMLreturn(caml_mpi_alloc_group(group));
}

value caml_mpi_group_incl(value group, value vranks)
{
  CAMLparam2(group, vranks);
  MPI_Group newgroup;
  int n = Wosize_val(vranks);
  int * ranks = caml_stat_alloc(n * sizeof(int));
  int i;

  for (i = 0; i < n; i++) ranks[i] = Int_field(vranks, i);
  MPI_Group_incl(Group_val(group), n, ranks, &newgroup);
  caml_stat_free(ranks);
  CAMLreturn(caml_mpi_alloc_group(newgroup));
}

value caml_mpi_group_excl(value group, value vranks)
{
  CAMLparam2(group, vranks);
  MPI_Group newgroup;
  int n = Wosize_val(vranks);
  int * ranks = caml_stat_alloc(n * sizeof(int));
  int i;

  for (i = 0; i < n; i++) ranks[i] = Int_field(vranks, i);
  MPI_Group_excl(Group_val(group), n, ranks, &newgroup);
  caml_stat_free(ranks);
  CAMLreturn(caml_mpi_alloc_group(newgroup));
}

static void caml_mpi_extract_ranges(value vranges,
                                    /*out*/ int * num,
                                    /*out*/ int (**rng)[3])
{
  CAMLparam1(vranges);
  CAMLlocal1(vrng);
  int n = Wosize_val(vranges);
  int (*ranges)[3] = caml_stat_alloc(n * sizeof(int[3]));
  int i;
  for (i = 0; i < n; i++) {
    caml_read_field(vranges, i, &vrng);
    ranges[n][0] = Int_field(vrng, 0);
    ranges[n][1] = Int_field(vrng, 1);
    ranges[n][2] = Int_field(vrng, 2);
  }
  *num = n;
  *rng = ranges;

  CAMLreturn0;
}

value caml_mpi_group_range_incl(value group, value vranges)
{
  CAMLparam2(group, vranges);
  int num;
  int (*ranges)[3];
  MPI_Group newgroup;
  caml_mpi_extract_ranges(vranges, &num, &ranges);
  MPI_Group_range_incl(Group_val(group), num, ranges, &newgroup);
  caml_stat_free(ranges);
  CAMLreturn(caml_mpi_alloc_group(newgroup));
}

value caml_mpi_group_range_excl(value group, value vranges)
{
  CAMLparam2(group, vranges);
  int num;
  int (*ranges)[3];
  MPI_Group newgroup;
  caml_mpi_extract_ranges(vranges, &num, &ranges);
  MPI_Group_range_excl(Group_val(group), num, ranges, &newgroup);
  caml_stat_free(ranges);
  CAMLreturn(caml_mpi_alloc_group(newgroup));
}


