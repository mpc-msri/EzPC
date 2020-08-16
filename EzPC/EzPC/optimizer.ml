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
open Tcenv
open Infer

let fresh_tac_var :unit -> var =
  let r = ref 0 in
  fun () -> r := !r + 1; { name = "__tac_var" ^ string_of_int !r; index = 0 }

type new_vars_t = (var * typ * expr) list

(*
 * turn an expression into 3-address code
 * input:
 *   g: input type environment, used to find the type of the introduced new variables
 *   vars: input list of new variables, threaded along, not actually used
 *   e: expression to apply 3-address code transformation to
 * output:
 *   g: output type environment that includes the introduced new variables
 *   vars: output list of introduced new variables
 *   e: expression in the 3-address code
 *)
let rec tac_expr (g:gamma) (vars:new_vars_t) (e:expr) :gamma * new_vars_t * expr =
  let aux (e:expr') :gamma * new_vars_t * expr' =
    match e with
    | Role _ -> g, vars, e
    | Const _ -> g, vars, e
    | Var _ -> g, vars, e
    | Unop (op, e, lopt) ->
       let g, vars, e = maybe_fresh_var g vars e in
       g, vars, Unop (op, e, lopt)
    | Binop (op, e1, e2, lopt) ->
       let g, vars, e1 = maybe_fresh_var g vars e1 in
       let g, vars, e2 = maybe_fresh_var g vars e2 in
       g, vars, Binop (op, e1, e2, lopt)
    | Conditional (e1, e2, e3, lopt) ->
       let g, vars, e1 = maybe_fresh_var g vars e1 in
       let g, vars, e2 = maybe_fresh_var g vars e2 in
       let g, vars, e3 = maybe_fresh_var g vars e3 in
       g, vars, Conditional (e1, e2, e3, lopt)
    | Array_read (e1, e2) ->
       let g, vars, e2 = maybe_fresh_var g vars e2 in
       g, vars, Array_read (e1, e2)
    | App (f, args) ->
       let g, vars, args = args |> List.fold_left (fun (g, vars, args) arg ->
                                       arg
                                       |> maybe_fresh_var g vars
                                       |> (fun (g, vars, arg) -> g, vars, args @ [arg])) (g, vars, [])
       in
       g, vars, App (f, args)
    | Subsumption (e, src, tgt) ->
       let g, vars, e = maybe_fresh_var g vars e in
       g, vars, Subsumption (e, src, tgt)
  in
  e.data |> aux |> (fun (g, vars, e') -> g, vars, { e with data = e' })

(*
 * this also turns the input expression to 3-address code
 * but introduces a fresh variable for the expression, in case it is already not a constant or an expression
 *)
and maybe_fresh_var (g:gamma) (vars:new_vars_t) (e:expr) :gamma * new_vars_t * expr =
  match e.data with
  | Role _ | Const _ | Var _ -> g, vars, e
  | _ ->
     let g, vars, e = tac_expr g vars e in
     let t = e |> typeof_expr g |> get_opt in
     if is_array_typ t then g, vars, e    (* don't create aliases of arrays *)
     else
       let x = fresh_tac_var () in
       add_local_binding g x t, vars @ [x, t, e], Var x |> mk_dsyntax ""

let rec tac_typ (g:gamma) (vars:new_vars_t) (t:typ) :gamma * new_vars_t * typ =
  match t.data with
  | Base _ -> g, vars, t
  | Array (quals, bt, e) ->
     let g, vars, bt = tac_typ g vars bt in
     let g, vars, e = maybe_fresh_var g vars e in
     g, vars, { t with data = Array (quals, bt, e) }
     
(*
 * get the declarations from the list of new variables
 * and return Seq (decls, s)
 *)
let new_vars_decls (s:stmt') (l:new_vars_t) :stmt' =
  (List.rev l) |> List.fold_left (fun s (x, t, e) ->
                      let s_decl = "Temporary variable for sub-expression on source location: " ^
                                     (Global.Metadata.sprint_metadata "" e.metadata)
                      in
                      Seq (Decl (t, Var x |> mk_dsyntax "", Some e) |> mk_dsyntax s_decl,
                           s |> mk_dsyntax "")) s

(*
 * apply the 3-address code transformation to the statement
 * also adds the introduced fresh variable declarations (via new_vars_decls)
 *)
let rec tac_stmt (g:gamma) (s:stmt) :gamma * stmt =
  let aux (g:gamma) (s:stmt') :gamma * stmt' =
    match s with
    | Decl (t, e_var, init_opt) ->
       let g, vars, init_opt =
         match init_opt with
         | None -> g, [], None
         | Some init -> init |> tac_expr g [] |> (fun (g, vars, init) -> g, vars, Some init)
       in
       add_local_binding g (get_var e_var) t, vars |> new_vars_decls (Decl (t, e_var, init_opt))
    | Assign (e1, e2) ->
       let g, vars, e1 = tac_expr g [] e1 in
       let g, vars, e2 = tac_expr g vars e2 in
       g, vars |> new_vars_decls (Assign (e1, e2))
    | Call (f, args) ->
       let g, vars, args = args |> List.fold_left (fun (g, vars, args) arg ->
                                       arg
                                       |> maybe_fresh_var g vars
                                       |> (fun (g, vars, arg) -> g, vars, args @ [arg])) (g, [], [])
       in
       g, vars |> new_vars_decls (Call (f, args))
    | For (quals, e_var, e1, e2, s) ->
       let g, vars, e1 = maybe_fresh_var g [] e1 in
       let g, vars, e2 = maybe_fresh_var g vars e2 in
       let g_body = {
           g with local_bindings = [get_var e_var,
                                    Base (Int32, Some Public) |> mk_dsyntax ""] |> push_stack g.local_bindings
         }
       in
       let _, s = tac_stmt g_body s in
       g, vars |> new_vars_decls (For (quals, e_var, e1, e2, s))
    | While (e, s) ->
       (* we don't tac the while expression, since lifting it out is unsound *)
       let g_body = { g with local_bindings = [] |> push_stack g.local_bindings } in
       let _, s = tac_stmt g_body s in
       g, While (e, s)
    | If_else (e, s_then, s_else_opt) ->
       let g, vars, e = maybe_fresh_var g [] e in
       let g_body = {
           g with local_bindings = [] |> push_stack g.local_bindings
         }
       in
       let _, s_then = tac_stmt g_body s_then in
       let s_else_opt = map_opt s_else_opt (fun s -> s |> tac_stmt g_body |> snd) in
       g, vars |> new_vars_decls (If_else (e, s_then, s_else_opt))
    | Return None -> g, Return None
    | Return (Some e) ->
       let g, vars, e = maybe_fresh_var g [] e in
       g, vars |> new_vars_decls (Return (Some e))
    | Seq (s1, s2) ->
       let g, s1 = tac_stmt g s1 in
       let g, s2 = tac_stmt g s2 in
       g, Seq (s1, s2)
    | Input (e_role, e_var, t) ->
       let g, vars, t = tac_typ g [] t in
       add_local_binding g (get_var e_var) t, vars |> new_vars_decls (Input (e_role, e_var, t))
    | Output (e_role, e, t) ->
       let g, vars, e = maybe_fresh_var g [] e in
       let g, vars, t =
         match t with
         | None -> g, vars, None
         | Some t -> tac_typ g vars t |> (fun (g, vars, t) -> g, vars, Some t)
       in
       g, vars |> new_vars_decls (Output (e_role, e, t))
    | Skip _ -> g, s
  in
  s.data |> aux g |> (fun (g, s') -> g, { s with data = s' })

let tac_global (g0:gamma) (d:global) :global * gamma =
  let aux (d:global') :global' * gamma =
    match d with
    | Fun (quals, fname, bs, body, ret_t) ->
       let g = { g0 with
                 top_level_functions = g0.top_level_functions @ [fname, (bs, ret_t)];
                 local_bindings = singleton_stack bs;
                 f_return_typ = Some ret_t
               }
       in
       let _, body = tac_stmt g body in
       Fun (quals, fname, bs, body, ret_t),
       { g0 with
         top_level_functions = g0.top_level_functions @ [fname, (bs, ret_t)];
         local_bindings = empty_stack;
         f_return_typ = None
       }
    | Extern_fun (quals, fname, bs, ret_t) ->
       Extern_fun (quals, fname, bs, ret_t),
       { g0 with
         top_level_functions = g0.top_level_functions @ [fname, (bs, ret_t)];
         local_bindings = empty_stack;
         f_return_typ = None
       }
    | Global_const (t, e_var, _) ->
       d,
       { g0 with top_level_consts = g0.top_level_consts @ [get_var e_var, t] }
  in
  aux d.data |> (fun (d', g) -> { d with data = d' }, g)
  
let tac_program (p:program) :program =
  p |> List.fold_left (fun (g, p) d ->
           let d, g = tac_global g d in
           g, p @ [d]) (empty_env, []) |> snd

(*
 * data structure that we maintain for the CSE pass
 * cse_list is the mapping of an lvalue to the expression that it currently evaluates to
 * subst_list is the list of pending substitutions (lvalue, var) that are applied to expressions
 *   before searching the CSE list for them
 * fdefs is the list of top-level functions (used to invalidate arrays as args in function calls)
 *
 * for example, consider: x = a + b; y = a + b; z = x + w; m = y + w;
 *
 *   with x = a + b, we add (x, a + b) to the cse_list, subst_list is currently empty
 *   with y = a + b, we apply pending substitutions to a + b, which returns a + b
 *                   and then search for a + b in the cse_list
 *                   we get a hit with x
 *                   so we transform the statement to y = x and add (y, x) to the subst_list
 *   with z = x + w, we apply pending substitutions to x + w
 *                   currently only pending substitution is y by x, which leaves x + w unchanged
 *                   we then search for x + w in the cse_list, and get no hit
 *                  so the statement remains z = x + w, and we add (z, x + w) to the cse_list
 *   with m = y + w, we first apply pending substitutions which transforms it to x + w
 *                   we then search for x + w in the cse_list and get a hit
 *                   note if we did not have the pending substitutions, this would not hit
 *
 * also the data structure is not at all optimized right now
 *   scanning and matching, for both searching and removing entries, is expensive
 *
 * another thing to note is that we don't really maintain scopes etc. at this point
 *   since by this time, all the variables should be unique already
 *)
type cse_t = {
    cse_list: (expr * expr) list;
    subst_list: (expr * expr) list;
    fdefs: (string * binder list) list;
  }
           
let add_cse_entry (tbl:cse_t) (t:expr * expr) :cse_t = { tbl with cse_list = t::tbl.cse_list }

let add_subst_entry (tbl:cse_t) (t:expr * expr) :cse_t = { tbl with subst_list = t::tbl.subst_list }

let lookup_cse_expr (tbl:cse_t) (e:expr) :expr * bool =  (* b indicates if there was a hit *)
  (* first apply the pending substitutions *)
  let e = tbl.subst_list |> List.fold_left (fun e (e_from, e_to) -> e |> subst_expr e_from e_to) e in
  if is_const e then e, true  (* return true so that the caller can add it to the subst list to enable const inlining *)
  else
    (* search *)
    let rhs_opt = tbl.cse_list |> List.find_opt (fun (_, rhs) -> eq_expr e rhs) in
    match rhs_opt with
    | None -> e, false  (* even if search failed, return the substituted expression *)
    | Some (lhs, _) -> lhs, true

(* remove all the entries for which lhs or rhs contain any variable in common with vars *)
let remove_cse_vars (tbl:cse_t) (vars:SSet.t) :cse_t =
  let pred = fun (e1, e2) -> SSet.inter vars (SSet.union (get_vars_in_expr e1) (get_vars_in_expr e2)) |> SSet.is_empty in
  {
    cse_list = tbl.cse_list |> List.filter pred;
    subst_list = tbl.subst_list |> List.filter pred;
    fdefs = tbl.fdefs
  }

let remove_cse_expr (tbl:cse_t) (e:expr) :cse_t = get_vars_in_expr e |> remove_cse_vars tbl

let rec cse_typ (tbl:cse_t) (t:typ) :typ =
  match t.data with
  | Base _ -> t
  | Array (quals, bt, e) -> e |> lookup_cse_expr tbl |> fst |> (fun e -> { t with data = Array (quals, bt |> cse_typ tbl, e) })

let rec cse_stmt (tbl:cse_t) (s:stmt) :cse_t * stmt =
  let out_tbl = s |> modifies tbl.fdefs |> remove_cse_vars tbl in
  let aux (tbl:cse_t) (s:stmt') :cse_t * stmt' =
    match s with
    | Decl (t, e_var, init_opt) ->
       let t = cse_typ tbl t in
       (match init_opt with
        | None -> tbl, Decl (t, e_var, None)
        | Some init ->
           (match lookup_cse_expr tbl init with
            | init, false -> (e_var, init) |> add_cse_entry out_tbl, Decl (t, e_var, Some init)
            | e, true -> (e_var, e) |> add_subst_entry out_tbl, Decl (t, e_var, Some e)))
    | Assign (e1, e2) ->
       (match lookup_cse_expr tbl e2 with
        | e2, false -> (e1, e2) |> add_cse_entry out_tbl, Assign (e1, e2)
        | e, true -> (e1, e) |> add_subst_entry out_tbl, Assign (e1, e))
    | Call (f, args) -> out_tbl, Call (f, args |> List.map (fun arg -> arg |> lookup_cse_expr tbl |> fst))
    | For (quals, e_var, e1, e2, s) ->
       let _, s = s |> cse_stmt out_tbl in
       out_tbl, For (quals, e_var, e1 |> lookup_cse_expr out_tbl |> fst, e2 |> lookup_cse_expr out_tbl |> fst, s)
    | While (e, s) ->
       let _, s = cse_stmt out_tbl s in
       out_tbl, While (e |> lookup_cse_expr out_tbl |> fst, s)
    | If_else (e, s_then, s_else_opt) ->
       let s_then = s_then |> cse_stmt tbl |> snd in
       let s_else_opt =
         match s_else_opt with
         | None -> None
         | Some s_else -> s_else |> cse_stmt tbl |> snd |> some
       in
       out_tbl, If_else (e |> lookup_cse_expr tbl |> fst, s_then, s_else_opt)
    | Return None -> out_tbl, Return None
    | Return (Some e) -> out_tbl, e |> lookup_cse_expr tbl |> fst |> (fun e -> Return (Some e))
    | Seq (s1, s2) ->
       let tbl, s1 = cse_stmt tbl s1 in
       let tbl, s2 = cse_stmt tbl s2 in
       tbl, Seq (s1, s2)
    | Input (e_role, e_var, t) -> out_tbl, Input (e_role, e_var, t |> cse_typ tbl)
    | Output (e_role, e, topt) -> out_tbl, e |> lookup_cse_expr tbl |> fst |> (fun e -> Output (e_role, e, map_opt topt (cse_typ tbl)))
    | Skip _ -> tbl, s
  in
  s.data |> aux tbl |> (fun (tbl, s') -> tbl, { s with data = s' })

let cse_global (tbl:cse_t) (d:global) :global * cse_t =
  let aux (d:global') :global' * cse_t =
    match d with
    | Fun (quals, fname, bs, body, ret_t) -> Fun (quals, fname, bs, body |> cse_stmt tbl |> snd, ret_t), tbl
    | Extern_fun (quals, fname, bs, ret_t) -> Extern_fun (quals, fname, bs, ret_t), tbl
    | Global_const (t, e_var, init) ->
       let init = init |> lookup_cse_expr tbl |> fst in
       Global_const (t, e_var, init), add_subst_entry tbl (e_var, init)
  in
  aux d.data |> (fun (d', out_tbl) -> { d with data = d' }, out_tbl)

let cse_program (p:program) :program =
  let empty_cse_tbl = {
      cse_list = [];
      subst_list = [];
      fdefs = p |> List.map (fun d -> match d.data with
                                      | Fun (_, f, bs, _, _) -> [f, bs]
                                      | Extern_fun (_, f, bs, _) -> [f, bs]
                                      | _ -> []) |> List.flatten
    }
  in
  
  p |> List.fold_left (fun (p, tbl) d -> let d, tbl = cse_global tbl d in p @ [d], tbl) ([], empty_cse_tbl) |> fst
