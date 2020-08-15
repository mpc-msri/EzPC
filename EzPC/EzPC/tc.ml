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

open Config
open Utils
open Global
open Ast
open Tcenv

type 'a result =
  | Type_error of (string * range)
  | Well_typed of 'a
                     
let bind (x:'a result) (f:'a -> 'b result) :'b result =
  match x with
  | Type_error s -> Type_error s
  | Well_typed t -> f t

type eresult = typ result
                  
let lookup_variable (g:gamma) (x:var) (r:range) :eresult =
  match Tcenv.lookup_variable g x with
  | None -> Type_error ("Variable " ^ x.name ^ " not found", r)
  | Some t -> Well_typed t

let lookup_fun (g:gamma) (f:string) (r:range) :(binder list * ret_typ) result =
  match lookup_fun g f with
  | None -> Type_error ("Function " ^ f ^ " not found", r)
  | Some bt -> Well_typed bt
  
let check_expected_typ (e:expr) (t:typ) (expected:typ) :unit result =
  let err = Type_error ("Expression " ^ (expr_to_string e) ^ " has type: " ^ (typ_to_string t) ^ ", expected: " ^ (typ_to_string expected), e.metadata)
  in
  match t.data, expected.data with
  | Base (UInt32, Some Public), Base (UInt64, Some Public) -> Well_typed ()
  | Base (Int32, Some Public), Base (Int64, Some Public) -> Well_typed ()
  | Base (bt1, l1), Base (bt2, l2)
       when ((bt1 = bt2) ||
               (bt1 = UInt32 && bt2 = Int32) ||
                 (bt1 = Int32 && bt2 = UInt32) ||
                   (bt1 = UInt64 && bt2 = Int64) ||
                     (bt1 = Int64 && bt2 = UInt64)) && l1 = l2 -> Well_typed ()
  | Array (quals, _, _), Array (quals_exp, _, _) ->
     let bt1, dims1 = get_array_bt_and_dimensions t in
     let bt2, dims2 = get_array_bt_and_dimensions expected in
     if bt1.data = bt2.data &&
          List.length dims1 = List.length dims2 &&
            not (List.mem Immutable quals && not (List.mem Immutable quals_exp))
     then Well_typed ()
     else err
  | _, _ -> err
       
let join_types_err (e1:expr) (e2:expr) (t1:typ) (t2:typ) (r:range) :eresult =
  Type_error ("Cannot join types " ^ typ_to_string t1 ^ " and " ^ typ_to_string t2 ^
                " for expressions " ^ expr_to_string e1 ^ " and " ^ expr_to_string e2, r)

let check_base_type (t:typ) (r:range) :unit result =
  match t.data with
  | Base _ -> Well_typed ()
  | _ -> Type_error ("Expected a base type, found: " ^ (typ_to_string t), r)
  
let check_base_type_and_label (e:expr) (t:typ) (l:label) :eresult =
  match t.data with
  | Base (_, Some lt) when l = lt -> Well_typed t
  | _ -> Type_error ("Expected a base type with label " ^ (label_to_string l) ^ " for expression " ^ (expr_to_string e) ^ ", found: " ^ (typ_to_string t), e.metadata)

let check_option_is_set (s:string) (lopt:'a option) (r:range) :'a result =
  if is_none lopt then Type_error ("Unset option in tc: " ^ s, r)
  else Well_typed (get_opt lopt)

let check_unop_label_is_consistent (e:expr) (op:unop) (l:label) :unit result =
  match l with
  | Public -> Well_typed ()
  | Secret l ->
     match op with
     | U_minus -> Type_error ("Unary minus should have been desugared: " ^ expr_to_string e, e.metadata)
     | Bitwise_neg when l = Boolean -> Well_typed ()
     | Not when l = Boolean -> Well_typed ()
     | _ -> Type_error ("Unary operator expected a boolean label: " ^ expr_to_string e, e.metadata)

(*
 * Bitwise operators are not supported with secret labels, ABY does bad things
 *)
let check_binop_label_is_consistent (e:expr) (op:binop) (l:label) :unit result =
  let err = Type_error ("Binop label is inconsistent: " ^ (expr_to_string e), e.metadata) in
  match l with
  | Public -> Well_typed ()
  | Secret l ->
     match op with
     | Sum | Sub | Div | Mod -> Well_typed ()
     | Mul when l = Arithmetic -> Well_typed ()
     | Greater_than | Less_than | Greater_than_equal | Less_than_equal | Is_equal when l = Boolean -> Well_typed ()
     | L_shift when l = Boolean -> Well_typed ()
     | R_shift_a when l = Boolean -> Well_typed ()
     | And when l = Boolean -> Well_typed ()
     | Or when l = Boolean -> Well_typed ()
     | Xor when l = Boolean -> Well_typed ()
     | R_shift_l when l = Boolean -> Well_typed ()
     | Bitwise_and when l = Boolean -> Well_typed ()
     | Bitwise_or when l = Boolean -> Well_typed ()
     | Bitwise_xor when l = Boolean -> Well_typed ()
     | _ -> err

let check_array_type_and_return_bt (e:expr) (t:typ) :eresult =
  match t.data with
  | Array (quals, bt, _) ->
     (match bt.data with
      | Base _ -> Well_typed bt
      | Array (_, bbt, e) -> Well_typed { bt with data = Array (quals, bbt, e) })
  | _ -> Type_error ("Expected an array type in expression " ^ (expr_to_string e) ^ ", found: " ^ (typ_to_string t), e.metadata)

let check_subsumption_type_and_label (e:expr) (t:typ) (l1:label) (l2:label) :eresult =
  bind (check_base_type_and_label e t l1) (fun _ ->
         if l1 = l2 || l2 = Public
         then Type_error ("Invalid labels in subsumption: " ^ (expr_to_string e), e.metadata)
         else
           match t.data with
           | Base (bt, _) -> Well_typed (Base (bt, Some l2) |> mk_syntax t.metadata)
           | _ -> failwith "check_subsumption_type_and_label: impossible branch")

let check_expected_int_typ (e:expr) (t:typ) (l:label) :unit result =
  let err = Type_error ("Expression " ^ (expr_to_string e) ^ " should have an int type with label " ^ label_to_string l ^
                          ", instead got: " ^ typ_to_string t, e.metadata) in
  match t.data with
  | Base (bt, Some lt) when (bt = UInt32 || bt = UInt64 || bt = Int32 || bt = Int64) && lt = l -> Well_typed ()
  | _ -> err

let check_expected_bool_typ (e:expr) (t:typ) (l:label) :unit result =
  let err = Type_error ("Expression " ^ (expr_to_string e) ^ " should have bool type with label " ^ label_to_string l ^
                          ", instead got: " ^ typ_to_string t, e.metadata) in
  match t.data with
  | Base (Bool, Some lt) when lt = l -> Well_typed ()
  | _ -> err

let check_non_void_ret_typ (f:string) (t:ret_typ) (r:range) :eresult =
  match t with
  | Typ t -> Well_typed t
  | Void _ -> Type_error ("Function " ^ f ^ " has a void return type", r)
       
let rec tc_expr (g:gamma) (e:expr) :eresult =
  match e.data with
  | Role r -> Well_typed (typeof_role e.metadata)

  | Const c -> Well_typed (typeof_const c e.metadata)

  | Var s -> lookup_variable g s e.metadata

  | Unop (op, e1, lopt) ->
     bind (check_option_is_set (expr_to_string e) lopt e.metadata) (fun l ->
            bind (check_unop_label_is_consistent e op l) (fun _ ->
                   bind (tc_expr g e1) (fun t1 ->
                          bind (match op with
                                | Bitwise_neg -> check_expected_int_typ e1 t1 l
                                | Not -> check_expected_bool_typ e1 t1 l
                                | _ -> Type_error ("Unexpected operator: " ^ unop_to_string op, e.metadata)) (fun _ -> Well_typed t1))))

  | Binop (op, e1, e2, lopt) ->
     bind (check_option_is_set (expr_to_string e) lopt e.metadata) (fun l ->
            bind (check_binop_label_is_consistent e op l) (fun _ ->
                   bind (tc_expr g e1) (fun t1 ->
                          bind (tc_expr g e2) (fun t2 ->
                                 match op with
                                 | Sum | Sub | Mul | Div | Mod | Pow | Bitwise_and | Bitwise_or | Bitwise_xor ->
                                    bind (check_expected_int_typ e1 t1 l) (fun _ ->
                                           bind (check_expected_int_typ e2 t2 l) (fun _ ->
                                                  match join_types t1 t2 with
                                                  | Some t -> Well_typed t
                                                  | None -> join_types_err e1 e2 t1 t2 e.metadata))
                                 | R_shift_a | L_shift | R_shift_l ->
                                    bind (check_expected_int_typ e1 t1 l) (fun _ ->
                                           bind (check_expected_int_typ e2 t2 Public) (fun _ -> Well_typed t1))
                                 | Greater_than | Less_than | Is_equal | Greater_than_equal | Less_than_equal -> 
                                    bind (check_expected_int_typ e1 t1 l) (fun _ ->
                                           bind (check_expected_int_typ e2 t2 l) (fun _ ->
                                                  match join_types t1 t2 with
                                                  | Some t -> Well_typed (Base (Bool, t |> get_bt_and_label |> snd) |> mk_syntax e.metadata)
                                                  | None   -> join_types_err e1 e2 t1 t2 e.metadata))
                                 | And | Or | Xor -> 
                                    bind (check_expected_bool_typ e1 t1 l) (fun _ ->
                                           bind (check_expected_bool_typ e2 t2 l) (fun _ -> Well_typed t1))
                                 | _ -> Type_error ("Unexpected operator: " ^ binop_to_string op, e.metadata)))))

  | Conditional (e1, e2, e3, Some Public) ->
     bind (tc_expr g e1) (fun t1 ->
            bind (check_expected_typ e1 t1 (Base (Bool, Some Public) |> mk_dsyntax "")) (fun _ ->
                   bind (tc_expr g e2) (fun t2 ->
                          bind (check_base_type t2 e2.metadata) (fun _ ->
                                 bind (tc_expr g e3) (fun t3 ->
                                        bind (check_base_type t3 e3.metadata) (fun _ ->
                                               match join_types t2 t3 with
                                               | Some t -> Well_typed t
                                               | None -> join_types_err e2 e3 t2 t3 e.metadata))))))


  | Conditional (e1, e2, e3, lopt) ->
     bind (check_option_is_set (expr_to_string e) lopt e.metadata) (fun _ ->
            bind (tc_expr g e1) (fun t1 ->
                   bind (check_expected_typ e1 t1 (Base (Bool, lopt) |> mk_dsyntax "")) (fun _ ->
                          bind (tc_expr g e2) (fun t2 ->
                                 bind (check_base_type_and_label e2 t2 (get_opt lopt)) (fun _ ->
                                        bind (tc_expr g e3) (fun t3 ->
                                               bind (check_base_type_and_label e3 t3 (get_opt lopt)) (fun _ ->
                                                      match join_types t2 t3 with
                                                      | Some t -> Well_typed t
                                                      | None -> join_types_err e2 e3 t2 t3 e.metadata)))))))

  | Array_read (e1, e2) ->
     bind (tc_expr g e1) (fun t1 ->
            bind (tc_expr g e2) (fun t2 ->
                   bind (check_array_type_and_return_bt e1 t1) (fun bt ->
                          bind (check_expected_int_typ e2 t2 Public) (fun _ -> Well_typed bt))))

  | App (f, args) ->
     bind (lookup_fun g f e.metadata) (fun (bs, ret_t) ->
            bind (if List.length args = List.length bs
                  then Well_typed ()
                  else Type_error ("Incorrect number of arguments in application: " ^ (expr_to_string e), e.metadata)) (fun _ ->
                   bind (List.fold_left2 (fun res (_, t) arg ->
                             bind res (fun _ ->
                                    bind (tc_expr g arg) (fun arg_t ->
                                           check_expected_typ arg arg_t t))) (Well_typed ()) bs args) (fun _ ->
                          bind (check_non_void_ret_typ f ret_t e.metadata) (fun ret_t ->
                                 if modifies_expr (g.top_level_functions |> List.map (fun (f, (bs, _)) -> (f, bs))) e = SSet.empty
                                 then Well_typed ret_t
                                 else Type_error ("Application in expression form should be pure: " ^ expr_to_string e, e.metadata)))))

  | Subsumption (e1, l1, l2) -> bind (tc_expr g e1) (fun t -> check_subsumption_type_and_label e t l1 l2)


type sresult = gamma result

let rec check_type_well_formedness (g:gamma) (t:typ) :unit result =
  let bitlen_err (n:int) = Type_error ("Incorrect bitlen, expected: " ^ (string_of_int (Config.get_bitlen ())) ^ ", found: " ^ (string_of_int n), t.metadata) in
  match t.data with
  | Base (_, None) -> Type_error ("Unlabeled type: " ^ (typ_to_string t), t.metadata)
  | Base (Bool, Some (Secret Arithmetic)) -> Type_error ("Bool type cannot be arithmetic shared: " ^ (typ_to_string t), t.metadata)
  | Base (UInt32, Some (Secret _))
  | Base (Int32, Some (Secret _)) -> if Config.get_bitlen () = 32 then Well_typed () else bitlen_err 32
  | Base (UInt64, Some (Secret _))
  | Base (Int64, Some (Secret _)) -> if Config.get_bitlen () = 64 then Well_typed () else bitlen_err 64
  | Base _ -> Well_typed ()
  | Array (_, bt, e) ->
     bind (check_type_well_formedness g bt) (fun _ ->
            bind (tc_expr g e) (fun t ->
                   bind (check_expected_int_typ e t Public) (fun _ -> Well_typed ())))

let check_ret_typ_well_formedness (g:gamma) (t:ret_typ) :unit result =
  match t with
  | Typ ({data = Array _; metadata = md }) -> Type_error ("Function return type cannot be an array type", md)
  | Typ t -> t |> check_type_well_formedness g
  | Void _ -> Well_typed ()
    
let check_input_role_and_type (r:role) (t:typ) :unit result =
  let err = Type_error ("For CLIENT or SERVER roles, the input type should be secret labeled, and for ALL, the input type should be public", t.metadata) in

  let l = t |> get_bt_and_label |> snd |> get_opt in
  let b =
    match r with
    | Both -> l = Public
    | _ -> is_secret_label l
  in
  if b then Well_typed () else err
  
let check_lvalue_and_typ (g:gamma) (e:expr) (t:typ) :unit result =
  bind (check_base_type t e.metadata) (fun _ ->
         match e.data with
         | Var x ->
            if g.top_level_consts |> List.assoc_opt x |> is_some
            then Type_error ("Cannot modify a top-level variable " ^ expr_to_string e, e.metadata)
            else Well_typed ()
         | Array_read (e1, _) ->
            bind (tc_expr g e1) (fun t1 ->
                   if get_typ_quals t1 |> List.mem Immutable
                   then Type_error ("Cannot modify a const array " ^ expr_to_string e, e.metadata)
                   else Well_typed ())
         | _ -> Type_error ("Expression " ^ expr_to_string e ^ " is not an lvalue", e.metadata))
  
let rec tc_stmt (g:gamma) (s:stmt) :sresult =
  match s.data with
  | Decl (t, e_var, Some init) when is_var e_var ->
     bind (check_type_well_formedness g t) (fun _ ->
            bind (check_base_type t t.metadata) (fun _ ->
                   bind (tc_expr g init) (fun t_init ->
                          bind (check_expected_typ init t_init t) (fun _ -> Well_typed (add_local_binding g (get_var e_var) t)))))

  | Decl (t, e_var, None) when is_var e_var ->
     bind (check_type_well_formedness g t) (fun _ -> Well_typed (add_local_binding g (get_var e_var) t))

  | Decl _ -> Type_error ("Non-variable declaration: " ^ (stmt_to_string s), s.metadata)
            
  | Call (f, args) ->
     bind (lookup_fun g f s.metadata) (fun (bs, ret_t) ->
            bind (if List.length args = List.length bs
                  then Well_typed ()
                  else Type_error ("Incorrect number of arguments in application: " ^ (stmt_to_string s), s.metadata)) (fun _ ->
                   bind (List.fold_left2 (fun res (_, t) arg ->
                             bind res (fun _ ->
                                    bind (tc_expr g arg) (fun arg_t ->
                                           check_expected_typ arg arg_t t))) (Well_typed ()) bs args) (fun _ ->
                          match ret_t with
                          | Typ _ -> Type_error ("Function " ^ f ^ " has a non-void return type", s.metadata)
                          | Void _ -> Well_typed g)))

  | Assign (e1, e2) ->
     bind (tc_expr g e1) (fun t1 ->
            bind (check_lvalue_and_typ g e1 t1) (fun _ ->
                   bind (tc_expr g e2) (fun t2 ->
                          bind (check_expected_typ e2 t2 t1) (fun _ -> Well_typed g))))

  | For (_, e_var, e1, e2, s) when is_var e_var ->    (* TODO: check for qualifier consistency? *)
     let x = get_var e_var in
     bind (tc_expr g e1) (fun t1 ->
            bind (check_expected_typ e1 t1 (Base (Int32, Some Public) |> mk_syntax e_var.metadata)) (fun _ ->
                   bind (tc_expr g e2) (fun t2 ->
                          bind (check_expected_typ e2 t2 t1) (fun _ ->
                                 let g_body = [x,
                                               Base (Int32, Some Public) |> mk_syntax e_var.metadata] |> push_local_scope g in
                                 bind (tc_stmt g_body s) (fun _ ->
                                        if SSet.mem x (modifies (g.top_level_functions |> List.map (fun (f, (bs, _)) -> (f, bs))) s)
                                        then Type_error ("Loop variable cannot be modified in the loop", s.metadata)
                                        else Well_typed g)))))
  | While (e, s) ->
     bind (tc_expr g e) (fun t ->
            bind (check_expected_bool_typ e t Public) (fun _ ->
                   let g_body = [] |> push_local_scope g in
                   bind (tc_stmt g_body s) (fun _ -> Well_typed g)))
  | For (_, e, _, _, _) -> Type_error ("For loop index must be a variable instead got: " ^ (expr_to_string e), e.metadata)

  | If_else (e, s_then, s_else_opt) ->
     bind (tc_expr g e) (fun t ->
            bind (check_expected_typ e t (Base (Bool, Some Public) |> mk_syntax e.metadata)) (fun _ ->
                   let g_body = [] |> push_local_scope g in
                   bind (tc_stmt g_body s_then) (fun _ ->
                          if is_none s_else_opt then Well_typed g
                          else bind (tc_stmt g_body (get_opt s_else_opt)) (fun _ -> Well_typed g))))

  | Return None ->
     bind (check_option_is_set (stmt_to_string s) g.f_return_typ s.metadata) (fun ret_t ->
            match ret_t with
            | Typ t -> Type_error ("Function has a non-void return type: " ^ typ_to_string t, t.metadata)
            | Void _ -> Well_typed g)
     
  | Return (Some e) ->
     bind (check_option_is_set (stmt_to_string s) g.f_return_typ e.metadata) (fun ret_t ->
            bind (tc_expr g e) (fun e_t ->
                   bind (check_non_void_ret_typ "" ret_t s.metadata) (fun ret_t ->
                          bind (check_expected_typ e e_t ret_t) (fun _ -> Well_typed g))))

  | Seq (s1, s2) -> bind (tc_stmt g s1) (fun g1 -> tc_stmt g1 s2)

  | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
     bind (check_type_well_formedness g t) (fun _ ->
            bind (check_input_role_and_type (get_role e_role) t) (fun _ -> Well_typed (add_local_binding g (get_var e_var) t)))

  | Input _ -> Type_error ("Input statement: " ^ (stmt_to_string s) ^ " is not well-typed", s.metadata)

  | Output (e_role, e2, topt) when is_role e_role ->
     bind (check_option_is_set (stmt_to_string s) topt s.metadata) (fun _ ->
            bind (tc_expr g e2) (fun t2 ->
                   bind (check_expected_typ e2 t2 (get_opt topt)) (fun _ -> Well_typed g)))

  | Output _ -> Type_error ("Output statement: " ^ (stmt_to_string s) ^ " is not well-typed", s.metadata)

  | Skip _ -> Well_typed g

let check_global_name_is_fresh (g:gamma) (s:string) (r:range) :unit result =
  if g.top_level_consts |> List.find_opt (fun (x, _) -> x.name = s) |> is_some ||
       g.top_level_functions |> List.assoc_opt s |> is_some
  then Type_error ("Name " ^ s ^ " is already defined in the global scope", r)
  else Well_typed ()

let rec check_no_return (s:stmt) (reason:string) :unit result =
  match s.data with
  | Decl _ -> Well_typed ()
  | Assign _ -> Well_typed ()
  | Call _ -> Well_typed ()
  | For (_, _, _, _, s) -> check_no_return s reason
  | While (_, s) -> check_no_return s reason
  | If_else (_, s_then, s_else_opt) ->
     bind (check_no_return s_then reason) (fun _ ->
            match s_else_opt with
            | None -> Well_typed ()
            | Some s -> check_no_return s reason)
  | Return _ -> Type_error (reason, s.metadata)
  | Seq (s1, s2) ->
     bind (check_no_return s1 reason) (fun _ -> check_no_return s2 reason)
  | Input _ -> Well_typed ()
  | Output _ -> Well_typed ()
  | Skip _ -> Well_typed ()

let check_fn_qualifiers (quals:qualifier list) (ret_t:ret_typ) (body:stmt option) :unit result =
  if quals |> List.mem Inline then
    bind (match ret_t with
          | Void _ -> Well_typed ()
          | Typ t -> Type_error ("Non-void functions cannot be inlined", t.metadata)
         ) (fun _ -> if quals |> List.mem Extern then Well_typed ()
                     else check_no_return (get_opt body) "Inline functions cannot have return statements")
  else Well_typed ()
            
let tc_global (g0:gamma) (d:global) :gamma result =
  match d.data with
  | Fun (quals, fname, bs, body, ret_t) ->
     bind (check_global_name_is_fresh g0 fname d.metadata) (fun _ ->
            let g_body = { g0 with local_bindings = singleton_stack [] } in
            bind (List.fold_left (fun res (x, t) ->
                      bind res (fun g ->
                             bind (check_type_well_formedness g t) (fun _ -> Well_typed (add_local_binding g x t)))
                    ) (Well_typed g_body) bs) (fun g ->
                   bind (check_ret_typ_well_formedness g ret_t) (fun _ ->
                          bind (tc_stmt { g with top_level_functions = (fname, (bs, ret_t))::g.top_level_functions;
                                                 f_return_typ = Some ret_t;
                                        } body) (fun _ ->
                                 bind (check_fn_qualifiers quals ret_t (Some body) ) (fun _ -> Well_typed (add_fun g0 d.data))))))
  | Extern_fun (quals, fname, bs, ret_t) ->
     bind (check_global_name_is_fresh g0 fname d.metadata) (fun _ ->
            let g_body = { g0 with local_bindings = singleton_stack [] } in
            bind (List.fold_left (fun res (x, t) ->
                      bind res (fun g ->
                             bind (check_type_well_formedness g t) (fun _ -> Well_typed (add_local_binding g x t)))
                    ) (Well_typed g_body) bs) (fun g ->
                   bind (check_ret_typ_well_formedness g ret_t) (fun _ ->
                                 bind (check_fn_qualifiers quals ret_t None ) (fun _ -> Well_typed (add_fun g0 d.data)))))
  | Global_const (t, e_var, init) when is_var e_var ->
     let x = get_var e_var in
     bind (check_base_type_and_label e_var t Public) (fun t ->
            bind (check_global_name_is_fresh g0 x.name e_var.metadata) (fun _ ->
                   bind (tc_expr g0 init) (fun t_init ->
                          bind (check_expected_typ e_var t_init t) (fun _ ->
                                 Well_typed (add_global_const g0 d.data)))))
  | Global_const (_, e, _) -> Type_error ("Global declaration not a var: " ^ expr_to_string e, d.metadata)

let tc_program (p:program) :unit result =
  if List.length p = 0 then Type_error ("Empty program!", dmeta "")
  else
    let rest, main = p |> List.rev |> (fun p -> p |> List.tl |> List.rev, p |> List.hd) in
    bind (rest |> List.fold_left (fun res d -> bind res (fun g -> tc_global g d)
                                 ) (Well_typed empty_env)) (fun g ->
           let err = Type_error ("The last definition should the main function with void return type and no arguments", main.metadata) in
           match main.data with
           | Fun (quals, fname, bs, body, ret_t) ->
              if fname = "main" && bs = [] && typ_of_ret_typ ret_t = None
              then bind (tc_global g main) (fun _ ->
                          bind (check_no_return body "Main function cannot have a return statement") (fun _ -> Well_typed ()))
              else err
           | _ -> err)
