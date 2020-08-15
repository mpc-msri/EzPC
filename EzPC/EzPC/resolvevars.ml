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
open Global
open Ast
open Stdint

type scope = (string * var) list

type gamma = scope stack

type 'a result =
  | Binding_error of (string * range)
  | Success of 'a
           
let bind (x:'a result) (f:'a -> 'b result) :'b result =
  match x with
  | Binding_error s -> Binding_error s
  | Success x -> f x

let check_fresh_in_current_scope (g:gamma) (x:var) (r:range) :unit result =
  let scope, _ = pop_stack g in
  if List.mem_assoc x.name scope
  then Binding_error ("Variable " ^ x.name ^ " is already defined in the current scope", r)
  else Success ()
    
let rec resolve_local_variable (g:gamma) (x:var) (r:range) :var result =
  if is_empty_stack g then Binding_error ("Cannot find variable: " ^ x.name, r)
  else
    let scope, rest = pop_stack g in
    match List.assoc_opt x.name scope with
    | None -> resolve_local_variable rest x r
    | Some x -> Success x

let rec resolve_vars_expr (g:gamma) (e:expr) :expr result =
  let rng = e.metadata in

  let aux (e:expr') :expr' result =
    match e with
    | Role _ -> Success e
    | Const _ -> Success e
    | Var x -> bind (resolve_local_variable g x rng) (fun x -> Success (Var x))
    | Unop (op, e, lopt) ->
       bind (resolve_vars_expr g e) (fun e -> Success (Unop (op, e, lopt)))
    | Binop (op, e1, e2, lopt) ->
       bind (resolve_vars_expr g e1) (fun e1 ->
              bind (resolve_vars_expr g e2) (fun e2 -> Success (Binop (op, e1, e2, lopt))))
    | Conditional (e1, e2, e3, lopt) ->
       bind (resolve_vars_expr g e1) (fun e1 ->
              bind (resolve_vars_expr g e2) (fun e2 ->
                     bind (resolve_vars_expr g e3) (fun e3 -> Success (Conditional (e1, e2, e3, lopt)))))
    | Array_read (e1, e2) ->
       bind (resolve_vars_expr g e1) (fun e1 ->
              bind (resolve_vars_expr g e2) (fun e2 -> Success (Array_read (e1, e2))))
    | App (f, el) ->
       bind (el |> List.fold_left (fun res e ->
                       bind res (fun el ->
                              bind (resolve_vars_expr g e) (fun e -> Success (e::el)))
                     ) (Success [])) (fun el -> Success (App (f, List.rev el)))
    | Subsumption (e, src, dst) ->
       bind (resolve_vars_expr g e) (fun e -> Success (Subsumption (e, src, dst)))
  in
  bind (aux e.data) (fun e' -> Success { e with data = e' })

let rec resolve_vars_typ (g:gamma) (t:typ) :typ result =
  let aux (t:typ') :typ' result =
    match t with
    | Base _ -> Success t
    | Array (quals, t, e) ->
       bind (resolve_vars_typ g t) (fun t ->
              bind (resolve_vars_expr g e) (fun e -> Success (Array (quals, t, e))))
  in
  bind (aux t.data) (fun t' -> Success { t with data = t' })

let resolve_vars_ret_typ (g:gamma) (t:ret_typ) :ret_typ result =
  match t with
  | Typ t -> bind (resolve_vars_typ g t) (fun t -> Success (Typ t))
  | Void md -> Success (Void md)
  
let get_fresh_index :unit -> int =
  let r = ref 0 in
  fun () -> r := !r + 1; !r
    
let gen_fresh_var (x:var) :var result = Success ({ x with index = get_fresh_index () })
    
let rec resolve_vars_stmt (g:gamma) (s:stmt) :(gamma * stmt) result =
  let rng = s.metadata in
  let aux (s:stmt') :(gamma * stmt') result =
    match s with
    | Decl (t, e, init_opt) when is_var e ->
       let x = get_var e in
       bind (resolve_vars_typ g t) (fun t ->
              bind (check_fresh_in_current_scope g x e.metadata) (fun _ ->
                     bind (gen_fresh_var x) (fun x ->
                            bind (match init_opt with
                                  | None -> Success None
                                  | Some init -> bind (resolve_vars_expr g init) (fun init -> Success (Some init))) (fun init_opt ->
                                   let scope, rest = pop_stack g in
                                   Success (((x.name, x)::scope) |> push_stack rest, Decl (t, { e with data = Var x }, init_opt))))))
    | Decl _ -> Binding_error ("Unexpected declaration: " ^ (stmt_to_string (s |> mk_dsyntax "")), rng)
    | Assign (e1, e2) ->
       bind (resolve_vars_expr g e1) (fun e1 ->
              bind (resolve_vars_expr g e2) (fun e2 -> Success (g, Assign (e1, e2))))
    | Call (f, el) ->
       bind (el |> List.fold_left (fun res e ->
                       bind res (fun el ->
                              bind (resolve_vars_expr g e) (fun e -> Success (e::el)))
                     ) (Success [])) (fun el -> Success (g, Call (f, List.rev el)))
    | For (quals, e, e1, e2, s) when is_var e ->
       let x = get_var e in
       bind (gen_fresh_var x) (fun x ->
              bind (resolve_vars_expr g e1) (fun e1 ->
                     bind (resolve_vars_expr g e2) (fun e2 ->
                            let g_body = [x.name, x] |> push_stack g in
                            bind (resolve_vars_stmt g_body s) (fun (_, s) -> Success (g, For (quals, { e with data = Var x }, e1, e2, s))))))
    | For _ -> Binding_error ("Unexpected for statement index: " ^ (stmt_to_string (s |> mk_dsyntax "")), rng)
    | While (e, s) ->
       bind (resolve_vars_expr g e) (fun e ->
              let g_body = [] |> push_stack g in
              bind (resolve_vars_stmt g_body s) (fun (_, s) -> Success (g, While (e, s))))
    | If_else (e, s_then, s_else_opt) ->
       bind (resolve_vars_expr g e) (fun e ->
              let g_body = [] |> push_stack g in
              bind (resolve_vars_stmt g_body s_then) (fun (_, s_then) ->
                     bind (match s_else_opt with
                           | None -> Success None
                           | Some s_else -> bind (resolve_vars_stmt g_body s_else) (fun (_, s_else) -> Success (Some s_else))) (fun s_else_opt ->
                            Success (g, If_else (e, s_then, s_else_opt)))))
    | Return None -> Success (g, Return None)
    | Return (Some e) -> bind (resolve_vars_expr g e) (fun e -> Success (g, Return (Some e)))
    | Seq (s1, s2) ->
       bind (resolve_vars_stmt g s1) (fun (g, s1) ->
              bind (resolve_vars_stmt g s2) (fun (g, s2) -> Success (g, Seq (s1, s2))))
    | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
       let x = get_var e_var in
       bind (resolve_vars_typ g t) (fun t ->
              bind (check_fresh_in_current_scope g x e_var.metadata) (fun _ ->
                     bind (gen_fresh_var x) (fun x ->
                            let scope, rest = pop_stack g in
                            Success (((x.name, x)::scope) |> push_stack rest, Input (e_role, { e_var with data = Var x }, t)))))
    | Input _ -> Binding_error ("Unexpected input statement: " ^ (stmt_to_string (s |> mk_dsyntax "")), rng)
    | Output (e_role, e, topt) when is_role e_role ->
       bind (resolve_vars_expr g e) (fun e ->
              bind (match topt with
                    | None -> Success None
                    | Some t -> bind (resolve_vars_typ g t) (fun t -> Success (Some t))) (fun topt -> Success (g, Output (e_role, e, topt))))
    | Output _ -> Binding_error ("Unexpected output statement: " ^ (stmt_to_string (s |> mk_dsyntax "")), rng)
    | Skip _ -> Success (g, s)
  in
  bind (aux s.data) (fun (g, s') -> Success (g, { s with data = s' }))

let resolve_vars_global (g0:gamma) (d:global) :(gamma * global) result =
  let rng = d.metadata in
  let aux (d:global') :(gamma * global') result =
    match d with
    | Fun (quals, fname, bs, body, ret_t) ->
       let g = [] |> push_stack g0 in
       bind (bs |> List.fold_left (fun res (x, t) ->
                       bind res (fun (g, bs) ->
                              bind (t |> resolve_vars_typ g) (fun t ->
                                     bind (x |> gen_fresh_var) (fun x ->
                                            let scope, rest = pop_stack g in
                                            Success ((scope @ [x.name, x]) |> push_stack rest,
                                                     bs @ [x, t]))))
                     ) (Success (g, []))) (fun (g, bs) ->
              bind (resolve_vars_ret_typ g ret_t) (fun ret_t ->
                     bind (resolve_vars_stmt g body) (fun (_, body) -> Success (g0, Fun (quals, fname, bs, body, ret_t)))))
    | Extern_fun (quals, fname, bs, ret_t) ->
       let g = [] |> push_stack g0 in
       bind (bs |> List.fold_left (fun res (x, t) ->
                       bind res (fun (g, bs) ->
                              bind (t |> resolve_vars_typ g) (fun t ->
                                     bind (x |> gen_fresh_var) (fun x ->
                                            let scope, rest = pop_stack g in
                                            Success ((scope @ [x.name, x]) |> push_stack rest,
                                                     bs @ [x, t]))))
                     ) (Success (g, []))) (fun (g, bs) ->
              bind (resolve_vars_ret_typ g ret_t) (fun ret_t -> Success (g0, Extern_fun (quals, fname, bs, ret_t))))
    | Global_const (t, e_var, init) when is_var e_var ->
       let x = get_var e_var in
       bind (resolve_vars_typ g0 t) (fun t ->
              bind (check_fresh_in_current_scope g0 x rng) (fun _ ->
                     bind (gen_fresh_var x) (fun x ->
                            bind (resolve_vars_expr g0 init) (fun init ->
                                   let scope, rest = pop_stack g0 in
                                   Success ((scope @ [x.name, x]) |> push_stack rest, Global_const (t, { e_var with data = Var x }, init))))))
    | Global_const (_, e_var, _) -> Binding_error ("Unexpected global constant declaration: " ^ expr_to_string e_var, rng)
                                     in
  bind (aux d.data) (fun (g, d') -> Success (g, { d with data = d' }))
  
let resolve_vars_program (p:program) :program result =
  let g = [] |> push_stack [] in
  bind (p |> List.fold_left (fun res d ->
                 bind res (fun (g, decls) ->
                        bind (resolve_vars_global g d) (fun (g, d) -> Success (g, decls @ [d])))) (Success (g, []))) (fun (_, p) -> Success p)
