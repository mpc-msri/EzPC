(*

Authors: Aseem Rastogi, Nishant Kumar, Mayank Rathee.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*)

open Utils
open Ast

type scope = (var * typ) list

type gamma = {
    top_level_consts: (var * typ) list;
    top_level_functions: (string * (binder list * ret_typ)) list;
    local_bindings: scope stack;
    f_return_typ: ret_typ option
  }

let add_local_binding (g:gamma) (x:var) (t:typ) :gamma =
  let scope, rest = pop_stack g.local_bindings in
  { g with local_bindings = ((x, t)::scope) |> push_stack rest }

let rec lookup_variable (g:gamma) (x:var) :typ option =
  if is_empty_stack g.local_bindings then List.assoc_opt x g.top_level_consts
  else
    let scope, rest = pop_stack g.local_bindings in
    match List.assoc_opt x scope with
    | None   -> lookup_variable { g with local_bindings = rest } x
    | Some t -> Some t

let lookup_fun (g:gamma) (f:string) :(binder list * ret_typ) option = List.assoc_opt f g.top_level_functions
              
let empty_env =
  {
    top_level_consts = [];
    top_level_functions = [];
    local_bindings = empty_stack;
    f_return_typ = None
  }

let enter_fun (g:gamma) (d:global') :gamma =
  match d with
  | Fun (quals, fname, bs, body, ret_t) ->
     { g with
       top_level_functions = (fname, (bs, ret_t))::g.top_level_functions;
       local_bindings = singleton_stack bs;
       f_return_typ = Some ret_t
     }
  | _ -> failwith "TcEnv::enter_fun should be called with a Fun global"

let add_fun (g:gamma) (d:global') :gamma =
  match d with
  | Fun (quals, fname, bs, _, ret_t) | Extern_fun (quals, fname, bs, ret_t) ->
     { g with
       top_level_functions = g.top_level_functions @ [fname, (bs, ret_t)];
       local_bindings = empty_stack;
       f_return_typ = None
     }
  | _ -> failwith "TcEnv::add_fun should be called with a Fun/Extern_Fun global"

let add_global_const (g:gamma) (d:global') :gamma =
  match d with
  | Global_const (t, e_var, init) ->
     { g with
       top_level_consts = g.top_level_consts @ [get_var e_var, t] }

  | _ -> failwith "TcEnv::add_global_const should be called with a Global"

let push_local_scope (g:gamma) (sc:scope) :gamma = { g with local_bindings = sc |> push_stack g.local_bindings }


(*
 * A best-case function that tries to return the label of the expression
 *
 * A successful Some result should not be interpreted as well-typedness
 *)
let join_labels (l1:label option) (l2:label option) :label option =
  match l1, l2 with
  | _, _ when l1 = l2 -> l1
  | Some _, None -> l1
  | None, Some _ -> l2
  | Some (Secret _), Some Public -> l1
  | Some Public, Some (Secret _) -> l2
  | _, _ -> None

let rec label_of_expr (g:gamma) (e:expr) :label option =
  match e.data with
  | Role _ -> Some Public
  | Const _ -> Some Public
  | Var x ->
     (match lookup_variable g x with
      | None -> None
      | Some t -> snd (get_bt_and_label t))
  (*
   * Is_equal is special, we allow x =_al y, but its return value is always boolean shared
   *)
  | Binop (Is_equal, _, _, Some (Secret _)) -> Some (Secret Boolean)
  | Binop (_, _, _, lopt) -> lopt
  | Unop (_, _, lopt) -> lopt
  | Conditional (_, e1, e2, lopt) ->
     if lopt = Some Public then join_labels (label_of_expr g e1) (label_of_expr g e2)
     else lopt
  | Array_read (e1, _) -> label_of_expr g e1
  | App (f, args) ->
     (match List.assoc_opt f g.top_level_functions with
      | None -> None
      | Some (_, Typ t) -> t |> get_bt_and_label |> snd
      | Some (_, Void _) -> Some Public)
  | Subsumption (_, _, tgt) -> Some tgt


(*
 * A best-case function that tries to compute the type of the expression
 *
 * A successful Some result should not be interpreted as well-typedness
 *)
let rec typeof_expr (g:gamma) (e:expr) :typ option =
  match e.data with
  | Role _ -> typeof_role e.metadata |> some
  | Const c -> typeof_const c e.metadata |> some
  | Var x -> lookup_variable g x
  | Unop (_, e, _) -> typeof_expr g e
  | Binop (op, e1, e2, lopt) ->
     (match op with
      | Sum | Sub | Mul | Div | Mod | Pow | Bitwise_and | Bitwise_or | Bitwise_xor ->
         map_opt (typeof_expr g e1) (fun t1 -> map_opt (typeof_expr g e2) (fun t2 -> join_types t1 t2)) |> double_opt |> double_opt
      | R_shift_a | L_shift | R_shift_l -> typeof_expr g e1
      | Greater_than | Less_than | Is_equal | Greater_than_equal | Less_than_equal ->
         Base (Bool, lopt) |> mk_syntax e.metadata |> some
      | And | Or | Xor -> typeof_expr g e1
      | _ -> failwith "infer::type_of_expr:This was supposed to be an exhaustive match")
  | Conditional (_, e2, e3, _) ->
     map_opt (typeof_expr g e2) (fun t2 -> map_opt (typeof_expr g e3) (fun t3 -> join_types t2 t3)) |> double_opt |> double_opt
  | Array_read (e1, _) ->
     map_opt (typeof_expr g e1) (fun t1 ->
               match t1.data with
               | Array (quals, bt, _) ->
                  (match bt.data with
                   | Base _ -> Some bt
                   | Array (_, bbt, e) -> Some { bt with data = Array (quals, bbt, e) })  (* if it's a partial array read, then propagate the qualifiers *)
               | _ -> None) |> double_opt
  | App (f, _) -> map_opt (List.assoc_opt f g.top_level_functions) (fun (_, ret_t) -> typ_of_ret_typ ret_t) |> double_opt
  | Subsumption (e, _, ltgt) ->
     map_opt (typeof_expr g e) (fun t ->
               match t.data with
               | Base (bt, _) -> Base (bt, Some ltgt) |> mk_syntax t.metadata |> some
               | _ -> None) |> double_opt
