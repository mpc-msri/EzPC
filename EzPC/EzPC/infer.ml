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
open Tcenv

let warn_label_inference_failed (e:expr') (r:range) :expr' =
  print_string ("WARNING: label inference failed for expression " ^ expr_to_string (e |> mk_dsyntax "")
                ^ Global.Metadata.sprint_metadata " " r ^ "\n");
  e

let rec infer_unop_label (g:gamma) (op:unop) (e:expr) (lopt:label option) (r:range) :expr' =
  let e = infer_op_labels_expr g e in
  match lopt with
  | Some _ -> Unop (op, e, lopt)
  | None ->
     match label_of_expr g e with
     | None -> warn_label_inference_failed (Unop (op, e, lopt)) r
     | Some Public -> Unop (op, e, Some Public)
     | Some _ -> Unop (op, e, Some (Secret Boolean))

and infer_binop_label (g:gamma) (op:binop) (e1:expr) (e2:expr) (lopt:label option) (rng:range) :expr' =
  let e1, e2 = infer_op_labels_expr g e1, infer_op_labels_expr g e2 in
  match lopt with
  | Some _ -> Binop (op, e1, e2, lopt)
  | None ->
     match label_of_expr g e1, label_of_expr g e2 with
     | None, _ -> warn_label_inference_failed (Binop (op, e1, e2, lopt)) rng
     | _, None -> warn_label_inference_failed (Binop (op, e1, e2, lopt)) rng
     | Some Public, Some Public -> Binop (op, e1, e2, Some Public)
     | Some l1, Some l2 ->
        let set_default_label (lopt:label option) :expr' =
          if l1 = Public then Binop (op, e1, e2, Some l2)
          else if l2 = Public then Binop (op, e1, e2, Some l1)
          else if l1 = l2 then Binop (op, e1, e2, Some l1)
          else if is_some lopt then Binop (op, e1, e2, Some (lopt |> get_opt))
          else warn_label_inference_failed (Binop (op, e1, e2, lopt)) rng
        in
        match op with
        | Sum | Sub | Div | Mod -> set_default_label None
        | Mul -> Binop (op, e1, e2, Some (Secret Arithmetic))
        | Pow -> Binop (op, e1, e2, Some Public)
        | Greater_than | Less_than | Greater_than_equal | Less_than_equal | Is_equal ->
           Binop (op, e1, e2, Some (Secret Boolean))
        | R_shift_a -> Binop (op, e1, e2, Some (Secret Boolean))
        | L_shift -> Binop (op, e1, e2, Some (Secret Boolean))
        | Bitwise_and -> Binop (op, e1, e2, Some (Secret Boolean))
        | Bitwise_or -> Binop (op, e1, e2, Some (Secret Boolean))
        | Bitwise_xor -> Binop (op, e1, e2, Some (Secret Boolean))
        | And -> Binop (op, e1, e2, Some (Secret Boolean))
        | Or -> Binop (op, e1, e2, Some (Secret Boolean))
        | Xor -> Binop (op, e1, e2, Some (Secret Boolean))
        | R_shift_l -> Binop (op, e1, e2, Some (Secret Boolean))
        | _ -> failwith "This was supposed to be an exhaustive match"
               
and infer_op_labels_expr (g:gamma) (e:expr) :expr =
  let aux (g:gamma) (e:expr') (rng:range) :expr' =
    match e with
    | Role _ -> e
    | Const _ -> e
    | Var _ -> e
    | Unop (op, e1, lopt) -> infer_unop_label g op e1 lopt rng
    | Binop (op, e1, e2, lopt) -> infer_binop_label g op e1 e2 lopt rng
    | Conditional (e1, e2, e3, lopt) ->
       let e1, e2, e3 = infer_op_labels_expr g e1, infer_op_labels_expr g e2, infer_op_labels_expr g e3 in
       if is_some lopt then Conditional (e1, e2, e3, lopt)
       else
         let lopt1, lopt2, lopt3 = label_of_expr g e1, label_of_expr g e2, label_of_expr g e3 in
         if is_none lopt1 || is_none lopt2 || is_none lopt3 then warn_label_inference_failed e rng
         else
           if lopt1 = Some Public then Conditional (e1, e2, e3, Some Public)
           else Conditional (e1, e2, e3, Some (Secret Boolean))
    | Array_read (e1, e2) -> Array_read (infer_op_labels_expr g e1, infer_op_labels_expr g e2)
    | App (f, args) -> App (f, List.map (infer_op_labels_expr g) args)
    | Subsumption _ -> warn_label_inference_failed e rng
  in
  aux g e.data e.metadata |> (fun e' -> { e with data = e' })

let rec infer_op_labels_typ (g:gamma) (t:typ) :typ =
  match t.data with
  | Base _ -> t
  | Array (quals, bt, e) -> { t with data = Array (quals, infer_op_labels_typ g bt, infer_op_labels_expr g e) }

let rec infer_typ_label (suff:string) (t:typ) :typ =
  match t.data with
  | Base (bt, Some _) -> t
  | Base (Bool, None) ->
     print_string ("Assigning default label Secret Boolean to: " ^ suff ^ "\n");
     { t with data = Base (Bool, Some (Secret Boolean)) }
  | Base (bt, None) ->
     print_string ("Assigning default label Secret Arithmetic to: " ^ suff ^ "\n");
     { t with data = Base (bt, Some (Secret Arithmetic)) }
  | Array (quals, t, e) -> { t with data = Array (quals, infer_typ_label suff t, e) }
                          
let rec infer_op_labels_stmt (g:gamma) (s:stmt) :stmt * gamma =
  let rng = s.metadata in
  let aux (g:gamma) (s:stmt') :stmt' * gamma =
    match s with
    | Decl (t, e_var, eopt) when is_var e_var ->
       let t = t
               |> infer_typ_label ((e_var |> get_var_name) ^ (Global.Metadata.sprint_metadata "" rng))
               |> infer_op_labels_typ g in
       Decl (t, e_var, map_opt eopt (infer_op_labels_expr g)),
       add_local_binding g (get_var e_var) t
    | Assign (e1, e2) ->
       Assign (infer_op_labels_expr g e1, infer_op_labels_expr g e2),
       g
    | Call (f, args) -> Call (f, List.map (infer_op_labels_expr g) args), g
    | For (quals, e_var, e1, e2, s) when is_var e_var ->
       let g_body = {
           g with local_bindings = [get_var e_var,
                                    Base (Int32, Some Public) |> mk_dsyntax ""] |> push_stack g.local_bindings
         }
       in
       For (quals, e_var, infer_op_labels_expr g e1, infer_op_labels_expr g e2,
            fst (infer_op_labels_stmt g_body s)),
       g
    | While (e, s) ->
       let g_body = { g with local_bindings = [] |> push_stack g.local_bindings } in
       While (infer_op_labels_expr g e, fst (infer_op_labels_stmt g_body s)),
       g
    | If_else (e, ts, es_opt) ->
       let g_body = {
           g with local_bindings = [] |> push_stack g.local_bindings
         }
       in
       If_else (infer_op_labels_expr g e, fst (infer_op_labels_stmt g_body ts),
                if is_none es_opt then None else Some (fst (infer_op_labels_stmt g_body (get_opt es_opt)))),
       g
    | Return eopt -> Return (map_opt eopt (infer_op_labels_expr g)), g
    | Seq (s1, s2) ->
       let s1, g = infer_op_labels_stmt g s1 in
       let s2, g = infer_op_labels_stmt g s2 in
       Seq (s1, s2), g
    | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
       let t = t |> infer_typ_label ((e_var |> get_var_name) ^ Global.Metadata.sprint_metadata "" rng) |> infer_op_labels_typ g in
       Input (e_role, e_var, t),
       add_local_binding g (get_var e_var) t
    | Output (e_role, e, t) when is_role e_role ->
       (*
        * We expect t to be None at this point
        * Output statements are not annotated with type in the source
        * And later on in insert_coercion we fill it up
        * May be we should assert this at this point?
        *)
       let t = map_opt (map_opt t (infer_typ_label (Global.Metadata.sprint_metadata "" rng))) (infer_op_labels_typ g) in
       Output (e_role, infer_op_labels_expr g e, t),
       g
    | _ -> s, g
  in
  aux g s.data |> (fun (s', g) -> { s with data = s' }, g)

let infer_ret_typ_label (t:ret_typ) :ret_typ =
  match t with
  | Typ t -> Typ (infer_typ_label (Global.Metadata.sprint_metadata "" t.metadata) t)
  | Void _ -> t
  
let infer_op_labels_global (g0:gamma) (d:global) :global * gamma =
  let aux (d:global') :global' * gamma =
    match d with
    | Fun (quals, fname, bs, body, ret_t) ->
       let g = { g0 with local_bindings = singleton_stack [] } in
       let g, bs = bs |> List.fold_left (fun (g, bs) (x, t) ->
                             let t = t |> infer_typ_label (x.name ^ (Global.Metadata.sprint_metadata "" t.metadata)) |> infer_op_labels_typ g in
                             add_local_binding g x t, bs @ [x, t]) (g, [])
       in
       let ret_t = infer_ret_typ_label ret_t in
       let g = { g with
                 top_level_functions = (fname, (bs, ret_t))::g0.top_level_functions;
                 f_return_typ = Some ret_t }
       in
       let body, _ = infer_op_labels_stmt g body in
       Fun (quals, fname, bs, body, ret_t),
       { g0 with
         top_level_functions = g0.top_level_functions @ [fname, (bs, ret_t)];
         local_bindings = empty_stack;
         f_return_typ = None }
    | Extern_fun (quals, fname, bs, ret_t) ->
       let g = { g0 with local_bindings = singleton_stack [] } in
       let _, bs = bs |> List.fold_left (fun (g, bs) (x, t) ->
                             let t = t |> infer_typ_label (x.name ^ (Global.Metadata.sprint_metadata "" t.metadata)) |> infer_op_labels_typ g in
                             add_local_binding g x t, bs @ [x, t]) (g, [])
       in
       let ret_t = infer_ret_typ_label ret_t in
       Extern_fun (quals, fname, bs, ret_t),
       { g0 with
         top_level_functions = g0.top_level_functions @ [fname, (bs, ret_t)];
         local_bindings = empty_stack;
         f_return_typ = None }
    | Global_const (t, e_var, init) ->
       (*
        * Note that global constants are assigned Public label in the parser itself
        *)
       let t = infer_op_labels_typ g0 t in
       let init = infer_op_labels_expr g0 init in
       Global_const (t, e_var, init),
       { g0 with
         top_level_consts = g0.top_level_consts @ [get_var e_var, t] }
  in
  aux d.data |> (fun (d', g) -> ({ d with data = d' }, g))
  
let infer_op_labels_program (p:program) :program =
  p |> List.fold_left (fun (g, p) d ->
           let d, g = infer_op_labels_global g d in
           g, p @ [d]) (empty_env, []) |> snd
    
let maybe_add_subsumption (g:gamma) (tgt:label option) (e:expr) :expr =
  let lopt = label_of_expr g e in
  match lopt, tgt with
  | Some Public, Some (Secret l) -> { e with data = Subsumption (e, Public, Secret l) }
  | Some (Secret l1), Some (Secret l2) when l1 <> l2 -> { e with data = Subsumption (e, Secret l1, Secret l2) }
  | _, _ -> e

let rec insert_coercions_expr (g:gamma) (e:expr) :expr =
  let e0 = e in
  let aux (g:gamma) (e:expr') :expr' =
    match e with
    | Role _ -> e
    | Const _ -> e
    | Var _ -> e
    | Unop (op, e, lopt) -> Unop (op, maybe_add_subsumption g lopt (insert_coercions_expr g e), lopt)
    | Binop (b, e1, e2, lopt) ->
       (match b with
        | R_shift_l | R_shift_a | L_shift ->  (* for these binops, the second argument is public, so don't need coercions there *)
         let e1 = maybe_add_subsumption g lopt (insert_coercions_expr g e1) in
         Binop (b, e1, e2, lopt)
        | _ ->
         let e1 = maybe_add_subsumption g lopt (insert_coercions_expr g e1) in
         let e2 = maybe_add_subsumption g lopt (insert_coercions_expr g e2) in
         Binop (b, e1, e2, lopt))
    | Conditional (e1, e2, e3, lopt) ->
       let label_branches = label_of_expr g e0 in
       let e1 = maybe_add_subsumption g lopt (insert_coercions_expr g e1) in
       let e2 = maybe_add_subsumption g label_branches (insert_coercions_expr g e2) in
       let e3 = maybe_add_subsumption g label_branches (insert_coercions_expr g e3) in
       Conditional (e1, e2, e3, lopt)
    | Array_read (e1, e2) -> Array_read (insert_coercions_expr g e1, insert_coercions_expr g e2)
    | App (f, args) ->
       let args = List.map (insert_coercions_expr g) args in
       (match List.assoc_opt f g.top_level_functions with
        | None -> App (f, args)
        | Some (bs, _) ->
           if List.length bs <> List.length args then App (f, args)
           else
             let ts = List.map snd bs in
             let args = List.map2 (fun e t -> maybe_add_subsumption g (snd (get_bt_and_label t)) e) args ts in
             App (f, args))
    | Subsumption _ -> e
  in
  aux g e.data |> (fun e' -> { e with data = e' })

let rec insert_coercions_stmt (g:gamma) (s:stmt) :stmt * gamma =
  let aux (g:gamma) (s:stmt') :(stmt' * gamma) =
    match s with
    | Decl (t, e_var, e_opt) when is_var e_var ->
       Decl (t, e_var, map_opt e_opt (fun e ->
                                 let e = insert_coercions_expr g e in
                                 maybe_add_subsumption g (snd (get_bt_and_label t)) e)),
       add_local_binding g (get_var e_var) t
    | Assign (e1, e2) ->
       let e1, e2 = insert_coercions_expr g e1, insert_coercions_expr g e2 in
       let lopt = label_of_expr g e1 in
       Assign (e1, maybe_add_subsumption g lopt e2),
       g
    | Call (f, args) ->
       let args = List.map (insert_coercions_expr g) args in
       (match List.assoc_opt f g.top_level_functions with
        | None -> Call (f, args), g
        | Some (bs, _) ->
           if List.length bs <> List.length args then Call (f, args), g
           else
             let ts = List.map snd bs in
             let args = List.map2 (fun e t -> maybe_add_subsumption g (snd (get_bt_and_label t)) e) args ts in
             Call (f, args), g)
    | For (quals, e_var, e1, e2, s) when is_var e_var ->
       let e1, e2 = insert_coercions_expr g e1, insert_coercions_expr g e2 in
       let g_body = [get_var e_var,
                     Base (Int32, Some Public) |> mk_dsyntax ""] |> push_local_scope g in
       For (quals, e_var, e1, e2, fst (insert_coercions_stmt g_body s)),
       g
    | While (e, s) ->
       let g_body = [] |> push_local_scope g in
       While (insert_coercions_expr g e, fst (insert_coercions_stmt g_body s)),
       g
    | If_else (e, t_s, e_opt) ->
       let g_body = [] |> push_local_scope g in
       If_else (insert_coercions_expr g e, fst (insert_coercions_stmt g_body t_s),
                map_opt e_opt (fun s_e -> fst (insert_coercions_stmt g_body s_e))),
       g
    | Return (Some e) ->
       e |> insert_coercions_expr g |> (fun e ->
        match g.f_return_typ with
        | None -> e
        | Some (Typ t) -> maybe_add_subsumption g (snd (get_bt_and_label t)) e
        | Some (Void _) -> e) |> (fun e -> Return (Some e), g)
    | Seq (s1, s2) ->
       let s1, g = insert_coercions_stmt g s1 in
       let s2, g = insert_coercions_stmt g s2 in
       Seq (s1, s2), g
    | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
       Input (e_role, e_var, t), add_local_binding g (get_var e_var) t
    | Output (e_role, e, None) when is_role e_role ->
       let e = insert_coercions_expr g e in
       Output (e_role, e, typeof_expr g e), g
    | _ -> s, g
  in
  aux g s.data |> (fun (s', g) -> { s with data = s' }, g)

let insert_coercions_global (g0:gamma) (d:global) :global * gamma =
  let aux (d:global') :global' * gamma =
    match d with
    | Fun (quals, fname, bs, body, ret_t) ->
       let g = enter_fun g0 d in
       let body, _ = insert_coercions_stmt g body in
       Fun (quals, fname, bs, body, ret_t), add_fun g0 d
    | Extern_fun (quals, fname, bs, ret_t) -> d, add_fun g0 d
    | Global_const (t, e_var, init) ->
       Global_const (t, e_var, init),
       add_global_const g0 d
  in
  aux d.data |> (fun (d', g) -> { d with data = d' }, g)
       
let insert_coercions_program (p:program) :program =
  p |> List.fold_left (fun (g, p) d ->
           let d, g = insert_coercions_global g d in
           g, p @ [d]) (empty_env, []) |> snd

let infer (p:program) :program = p |> infer_op_labels_program |> insert_coercions_program
