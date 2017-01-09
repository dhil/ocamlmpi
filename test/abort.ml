(* comm_size, comm_rank *)

let size = Mpi.comm_size Mpi.comm_world
let myrank = Mpi.comm_rank Mpi.comm_world

let _ =
  Printf.printf "%d: comm_size = %d" myrank size; print_newline()

(* Barrier *)

let _ = Mpi.barrier Mpi.comm_world

(* Abort comm_world *)
let _ =
  if myrank = 0 then Mpi.abort Mpi.comm_world 42 (* initiate hard exit *)
  else ()

let _ = Mpi.barrier Mpi.comm_world (* Shouldn't be reached by process 0 *)
