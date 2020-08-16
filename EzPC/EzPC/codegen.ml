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

open Printf
open Char
open Stdint
open Ast
open Codegenast
open Utils
open Global
open Tcenv

open Codegenlib

let o_role (r:role) :comp =
  match r with
  | Server -> o_str "SERVER"
  | Client -> o_str "CLIENT"
  | Both   -> o_str "ALL"

let o_var (x:var) :comp = o_str x.name (* o_str (x.name ^ (string_of_int x.index)) *)

let o_punop :unop -> comp = function
  | U_minus -> o_str "-"
  | Bitwise_neg -> o_str "~"
  | Not -> o_str "!"
                        
let o_pbinop :binop -> comp = function
  | Sum          -> o_str "+"
  | Sub          -> o_str "-"
  | Mul          -> o_str "*"
  | Div          -> o_str "/"
  | Mod          -> o_str "%"
  | Pow          -> failwith "Pow is not an infix operator, so o_pbinop can't handle it"
  | Greater_than -> o_str ">"
  | Less_than    -> o_str "<"
  | Is_equal     -> o_str "=="
  | Greater_than_equal -> o_str ">="
  | Less_than_equal -> o_str "<="
  | R_shift_a    -> o_str ">>"
  | L_shift      -> o_str "<<"
  | Bitwise_and  -> o_str "&"
  | Bitwise_or   -> o_str "|"
  | Bitwise_xor  -> o_str "^"
  | And          -> o_str "&&"
  | Or           -> o_str "||"
  | Xor          -> o_str "^"
  | R_shift_l    -> o_str ">>"

(*
 * ABY doesn't like circ->PutINVGate where circ:Circuit*, so we need to coerce it to BooleanCircuit*
 * Providing a more generic flag
 *)
let o_slabel_maybe_coerce (coerce:bool) (sl:secret_label) :comp =
  match sl with
  | Arithmetic -> o_str "acirc"
  | Boolean    ->
    let c = if Config.get_bool_sharing_mode () = Config.Yao then o_str "ycirc"
            else o_str "bcirc" in
    if coerce then o_paren (seq (o_str "(BooleanCircuit *) ") c) else c
                  
let o_slabel :secret_label -> comp = o_slabel_maybe_coerce false

let o_hd_and_args (head:comp) (args:comp list) :comp =
  match args with
  | []     -> seq head (seq (o_str "(") (o_str ")"))
  | hd::tl ->
     let c = seq head (seq (o_str "(") hd) in
     let c = List.fold_left (fun c arg -> seq c (seq (o_str ", ") arg)) c tl in
     seq c (o_str ")")

let o_app (head:comp) (args:comp list) :comp = o_hd_and_args head args

let o_cbfunction_maybe_coerce (coerce:bool) (l:secret_label) (f:comp) (args:comp list) :comp =
  o_app (seq (o_slabel_maybe_coerce coerce l) (seq (o_str "->") f)) args

let o_cbfunction :secret_label -> comp -> comp list -> comp = o_cbfunction_maybe_coerce false

let o_sunop (l:secret_label) (op:unop) (c:comp) :comp =
  let c_op =
    match op with
    | U_minus -> failwith "Codegen: unary minus is not being produced by lexer or parser right now."
    | Bitwise_neg 
    | Not -> o_str "PutINVGate"
  in
  o_cbfunction_maybe_coerce true l c_op [c]
  
let o_sbinop (l:secret_label) (op:binop) (c1:comp) (c2:comp) :comp =
  let aux (s:string) (coerce:bool) :comp = o_cbfunction_maybe_coerce coerce l (o_str s) [c1; c2] in
  let err (s:string) = failwith ("Operator: " ^ s ^ " should have been desugared") in
  match op with
  | Sum                -> aux "PutADDGate" false
  | Sub                -> aux "PutSUBGate" false
  | Mul                -> aux "PutMULGate" false
  | Greater_than       -> aux "PutGTGate" false
  | Div                -> err "DIV"
  | Mod                -> err "MOD"
  | Less_than          -> err "LT"
  | Is_equal           -> err "EQ"
  | Greater_than_equal -> err "GEQ"
  | Less_than_equal    -> err "LEQ"
  | R_shift_a          -> o_app (o_str "arithmetic_right_shift") [o_slabel l; c1; c2]
  | L_shift            -> o_app (o_str "left_shift") [o_slabel l; c1; c2]
  | Bitwise_and        -> aux "PutANDGate" false
  | Bitwise_or         -> aux "PutORGate" true
  | Bitwise_xor        -> aux "PutXORGate" false
  | And                -> aux "PutANDGate" false
  | Or                 -> aux "PutORGate" true
  | Xor                -> aux "PutXORGate" false
  | R_shift_l          -> o_app (o_str "logical_right_shift") [o_slabel l; c1; c2]
  | Pow                -> failwith ("Codegen cannot handle this secret binop: " ^ binop_to_string op)
               
let o_pconditional (c1:comp) (c2:comp) (c3:comp) :comp =
  seq c1 (seq (o_str " ? ") (seq c2 (seq (o_str " : ") c3)))
  
let o_sconditional (l:secret_label) (c_cond:comp) (c_then:comp) (c_else:comp) :comp =
  o_cbfunction l (o_str "PutMUXGate") [c_then; c_else; c_cond]

let o_subsumption (src:label) (tgt:secret_label) (t:typ) (arg:comp) :comp =
  match src with
    | Public -> 
       let fn =
         o_str (
             match t.data with
             | Base (UInt32, _) | Base (Int32, _) -> "put_cons32_gate"
             | Base (UInt64, _) | Base (Int64, _) -> "put_cons64_gate"
             | Base (Bool, _) -> "put_cons1_gate"
             | _ -> failwith ("codegen:Subsumption node with an unexpected typ: " ^ (typ_to_string t)))
       in
       o_app fn [o_slabel tgt; arg]
    | Secret Arithmetic ->
       let fn_name = if Config.get_bool_sharing_mode () = Config.Yao then "PutA2YGate" else "PutA2BGate" in
       o_cbfunction tgt (o_str fn_name) [arg]
    | Secret Boolean ->
       let fn_name, circ_arg =
         if Config.get_bool_sharing_mode () = Config.Yao then "PutY2AGate", "bcirc"
         else "PutB2AGate", "ycirc"
       in
       o_cbfunction tgt (o_str fn_name) [arg; o_str circ_arg]

let o_basetyp (t:base_type) :comp =
  match t with
  | UInt32 -> o_str "uint32_t"
  | UInt64 -> o_str "uint64_t"
  | Int32  -> o_str "int32_t"
  | Int64  -> o_str "int64_t"
  | Bool   -> o_str "uint32_t"

let rec o_secret_binop (g:gamma) (op:binop) (sl:secret_label) (e1:expr) (e2:expr) :comp =
  (*
  * For some ops like shifts, type of whole expression is defined by 1st arg and not join of 1st and 2nd arg.
  *)
  let t1 = e1 |> typeof_expr g |> get_opt in
  let is_signed =
    match t1.data with
    | Base (Int32, _) | Base (Int64, _) -> true
    | _ -> false
  in
  let fn_name (s:string) =
    ((if is_signed then "signed" else "unsigned") ^ s ^
       (if sl = Arithmetic then "al" else "bl")) |> some
  in
  let app_opt =
    let fn_name_opt =
      match op with
      | Div -> fn_name "div"
      | Mod -> fn_name "mod"
      | Is_equal -> fn_name "equals"
      | Less_than -> fn_name "lt"
      | Greater_than_equal -> fn_name "geq"
      | Less_than_equal -> fn_name "leq"
      | Greater_than when is_signed -> fn_name "gt"
      | R_shift_a when is_signed -> fn_name "arshift"
      | _ -> None
    in
    map_opt fn_name_opt (fun s -> App (s, [e1; e2]))
  in
  match app_opt with
  | Some app -> o_expr g (app |> mk_dsyntax "")
  | None -> o_sbinop sl op (o_expr g e1) (o_expr g e2)

and o_expr (g:gamma) (e:expr) :comp =
  let o_expr = o_expr g in
  let o_codegen_expr = o_codegen_expr g in
  match e.data with
  | Role r -> o_role r

  | Const (Int32C n) -> seq (o_str (" (int32_t)")) (o_int32 n)

  | Const (Int64C n) -> seq (o_str (" (int64_t)")) (o_int64 n)

  | Const (UInt32C n) -> seq (o_str (" (uint32_t)")) (o_uint32 n)

  | Const (UInt64C n) -> seq (o_str (" (uint64_t)")) (o_uint64 n)

  | Const (BoolC b) -> o_bool b
    
  | Var s -> o_var s

  | Unop (op, e, Some Public) -> seq (o_punop op) (seq o_space (o_expr e))
  
  | Unop (op, e, Some (Secret s)) -> o_sunop s op (o_expr e)

  | Binop (op, e1, e2, Some Public) ->
     o_paren (match op with
              | R_shift_l -> o_app (o_str "public_lrshift") [o_expr e1; o_expr e2]
              | Pow -> o_app (o_str "pow") [o_expr e1; o_expr e2]
              | _ -> seq (o_expr e1) (seq o_space (seq (o_pbinop op) (seq o_space (o_expr e2)))))

  | Binop (op, e1, e2, Some (Secret s)) -> o_secret_binop g op s e1 e2

  | Conditional (e1, e2, e3, lopt) -> o_codegen_expr (Conditional_codegen (Base_e e1, Base_e e2, Base_e e3, (get_opt lopt)))

  | Array_read (e1, e2) -> seq (o_expr e1) (seq (o_str "[") (seq (o_expr e2) (o_str "]")))

  | App (f, args) -> o_app (o_str f) (List.map o_expr args)

  | Subsumption (e, l1, Secret l2) -> o_subsumption l1 l2 (typeof_expr g e |> get_opt) (o_expr e)

  | _ -> failwith "o_expr: impossible branch"

and o_codegen_expr (g:gamma) (e:codegen_expr) :comp =
  let o_expr = o_expr g in
  let o_codegen_expr = o_codegen_expr g in
  let get_bitlen bt = 
    match bt with
    | Bool -> o_uint32 (Uint32.of_int 1)
    | _ -> o_str "bitlen"
  in
  match e with
  | Base_e e -> o_expr e
              
  | Input_g (r, sl, s, bt) -> o_cbfunction sl (o_str "PutINGate") [o_str s.name; get_bitlen bt; o_role r]

  | Dummy_g (sl, bt) -> o_cbfunction sl (o_str "PutDummyINGate") [get_bitlen bt]
    
  | Output_g (r, sl, e) -> o_cbfunction sl (o_str "PutOUTGate") [o_codegen_expr e; o_role r]

  | Clear_val (e, bt) -> seq (o_codegen_expr e) (seq (o_str "->get_clear_value<") (seq (o_basetyp bt) (o_str ">()")))

  | Conditional_codegen (e1, e2, e3, l) ->
     let c1, c2, c3 = o_codegen_expr e1, o_codegen_expr e2, o_codegen_expr e3 in
     (match l with
      | Public -> o_pconditional c1 c2 c3
      | Secret sl -> o_sconditional sl c1 c2 c3)

  | App_codegen_expr (f, el) -> o_app (o_str f) (List.map o_codegen_expr el)
                                      
let o_typ (t:typ) :comp =
  match t.data with
  | Base (bt, Some (Secret _)) -> o_str "share*"
  | Base (bt, _) -> o_basetyp bt
  | Array (quals, _, _) -> seq (if quals |> List.mem Immutable then o_str "const " else o_null) (o_str "auto")

let o_ret_typ (t:ret_typ) :comp =
  match t with
  | Typ t -> o_typ t
  | Void _ -> o_str "void"
             
let o_array_init (g:gamma) (t:typ) :comp =
  let t, l = get_array_bt_and_dimensions t in
  let s = seq (o_str "make_vector<") (seq (o_typ t) (o_str ">")) in
  o_app s (List.map (o_expr g) l)

let o_for (index:comp) (lower:comp) (upper:comp) (body:comp) :comp =
  let init = seq (o_str "for (uint32_t ") (seq index (seq (o_str " = ") lower)) in
  let term = seq index (seq (o_str " < ") upper) in
  let incr = seq index (o_str "++)") in
  let header = seq init (seq (o_str "; ") (seq term (seq (o_str "; ") (seq incr (o_str "{"))))) in
  seq header (seq o_newline (seq body (seq o_newline (o_str "}"))))

let o_while (guard:comp) (body:comp) :comp =
  seq (o_str "while (") (seq guard (seq (o_str ") {\n") (seq body (o_str "\n}\n"))))
  
let o_ite (c_if:comp) (c_then:comp) (c_else_opt:comp option) :comp =
  let c_if_then = seq (o_str "if (") (seq c_if (seq (o_str ") {") (seq o_newline (seq c_then (seq o_newline (o_str "}")))))) in
  if is_none c_else_opt then c_if_then
  else seq c_if_then (seq (o_str " else {") (seq o_newline (seq (get_opt c_else_opt) (seq o_newline (o_str "}")))))

let o_assign (lhs:comp) (rhs:comp) :comp = seq lhs (seq (o_str " = ") rhs)

let o_decl (c_t:comp) (c_varname:comp) (init_opt:comp option) :comp =
  let c = seq c_t (seq o_space c_varname) in
  if is_none init_opt then c
  else seq c (seq (o_str " = ") (get_opt init_opt))
                                         
let get_fresh_var :(unit -> string) =
  let r = ref 0 in
  fun _ -> let s = "__tmp__fresh_var_" ^ (string_of_int !r) in r := !r + 1; s

let o_comment (s:string) :comp = seq (o_str "/* ") (seq (o_str s) (o_str " */"))

(*
 * Reference to keep the files opened for writing shares, so that we can close them
 * Bad! Make is part of the monad
 *)
let out_files :(string list) ref = ref []

let rec o_stmt (g:gamma) (s:stmt) :comp * gamma =
  match s.data with
  | Decl (t, e, init_opt) ->
     let o_init =
       match init_opt with
       | Some e -> Some (o_expr g e)
       | None -> if is_array_typ t then Some (o_array_init g t) else None
     in
     let comment =
       let c = Global.get_comment s.metadata in
       if c = "" then o_null else o_comment c
     in
     seq comment (seq o_newline (o_with_semicolon (o_decl (o_typ t) (o_expr g e) o_init))),
     add_local_binding g (get_var e) t
    
  | Assign (e1, e2) -> o_codegen_stmt g (Assign_codegen (Base_e e1, Base_e e2))

  | Call (f, args) -> o_with_semicolon (o_app (o_str f) (List.map (o_expr g) args)), g

  | For (_, e1, e2, e3, s) -> o_codegen_stmt g (For_codegen (Base_e e1, Base_e e2, Base_e e3, Base_s s))

  | While (e, s) -> o_while (o_expr g e) (o_stmt ([] |> push_local_scope g) s |> fst), g

  | If_else (e, s_then, s_else_opt) ->
     o_codegen_stmt g (If_codegen (Public, Base_e e, Base_s s_then, map_opt s_else_opt (fun s -> Base_s s)))

  | Return eopt -> o_with_semicolon (seq (o_str "return ")
                                         (if is_none eopt then o_null else eopt |> get_opt |> o_expr g)), g
    
  | Seq (s1, s2) -> o_codegen_stmt g (Seq_codegen (Base_s s1, Base_s s2))

  | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
     let rng = s.metadata in
     let r, x = get_role e_role, get_var e_var in
     let is_arr = is_array_typ t in
     
     (* bt is the base type and l label *)
     let bt, l = get_bt_and_label t |> (fun (bt, l) -> get_unsigned bt, l) in
     let l = get_opt l in

     (* list of dimensions, if an array else empty *)
     let el = if is_arr then snd (get_array_bt_and_dimensions t) else [] in
     
     (* the share typed variable *)
     let decl = Base_s (Decl (t, Var x |> mk_dsyntax "", None) |> mk_dsyntax "") in

     (* temporary variable in which we will cin *)
     let tmp_var_name = {
         name = "__tmp_in_" ^ x.name;
         index = 0
       }
     in
     let s_decl_tmp =
       "Variable to read the clear value corresponding to the input variable " ^ x.name ^
         " at " ^ (Global.Metadata.sprint_metadata "" rng)
     in
     let decl_tmp = Base_s (Decl (Base (bt, Some Public) |> mk_dsyntax "",
                                  Var tmp_var_name |> mk_dsyntax "", None) |> mk_dsyntax s_decl_tmp) in
     
     (* expression that we will initialize each optional loop iteration *)
     let assgn_left =
       Base_e (snd (List.fold_left (fun (i, e) _ ->
                        let i_var = { name = "i" ^ (string_of_int i); index = 0; } in
                        i + 1, Array_read (e, Var i_var |> mk_dsyntax "") |> mk_dsyntax ""
                      ) (0, Var x |> mk_dsyntax "") el))
     in

     (* conditional expression for role == r *)
     let r_cmp =
       let role_var = { name = "role"; index = 0 } in
       Base_e (Binop (Is_equal, Var role_var |> mk_dsyntax "", Role r |> mk_dsyntax "", Some Public) |> mk_dsyntax "")
     in
     
     (* this is the innermost loop body *)
     let base_init =
       (* if role == r then cin into the temporary variable *)
       let cin = Cin ("cin", Base_e (Var tmp_var_name |> mk_dsyntax "") , bt) in

       if is_secret_label l then
         let sl = get_secret_label l in
         let cin = If_codegen (Public, r_cmp, cin, None) in
         (* add an input gate *)
         let assgn = Assign_codegen (assgn_left,
                                     Conditional_codegen (r_cmp,
                                                          Input_g (r, sl, tmp_var_name, bt),
                                                          Dummy_g (sl, bt), Public))
         in
         Seq_codegen (cin, assgn)
       else
         let assgn = Assign_codegen (assgn_left,
                                     Base_e (Var tmp_var_name |> mk_dsyntax ""))
         in
         Seq_codegen (cin, assgn)
     in
     
     (* these are the for loops on the outside, note fold_right *)
     let loops =
       snd (List.fold_right (fun e (i, s) ->
                let i_var = { name = "i" ^ (string_of_int i); index = 0; } in
                i - 1,
                For_codegen (Base_e (Var i_var |> mk_dsyntax ""),
                             Base_e (Const (UInt32C (Uint32.of_int 0)) |> mk_dsyntax ""),
                             Base_e e,
                             s)
              ) el (List.length el - 1, base_init))
     in

     (* adding a print statement for the input *)
     (*
      * ideally we want cout for a string, can be added easily to codegenast.ml,
      * but for now abusing the codegen for variables
      *)
     let print_input_message =
       let x = { name = "\"Input " ^ x.name ^ ":\""; index = 0 } in
       let cout_stmt = Cout ("cout", Base_e (Var x |> mk_dsyntax ""), bt) in

       (* is_secret_label l is also a proxy for codegen ABY, since labels are erased already if codegen CPP *)
       if is_secret_label l then
         If_codegen (Public, r_cmp, cout_stmt, None)
       else
         cout_stmt
     in
     
     (* stitch *)
     o_codegen_stmt g (Seq_codegen (decl, Seq_codegen (Seq_codegen (print_input_message, decl_tmp), loops)))

  | Output (e_role, e, Some t) when is_role e_role ->
     let r = get_role e_role in
     let bt, l = get_bt_and_label t in

     if not (l |> get_opt |> is_secret_label) then
      let print_output_msg =
        let msg = Var { name = "\"Value of " ^ (expr_to_string e) ^ ":\""; index = 0 } |> mk_dsyntax "" in
        Cout ("cout", Base_e msg, bt)
      in
      o_codegen_stmt g (Seq_codegen (print_output_msg, Cout ("cout", Base_e e, bt)))
     else
      let print_output_msg =
        let msg = Var { name = "\"Value of " ^ (expr_to_string e) ^ ":\""; index = 0 } |> mk_dsyntax "" in
        App_codegen ("add_print_msg_to_output_queue", [Base_e (Var { name = "out_q"; index = 0 } |> mk_dsyntax "");
                                                       Base_e msg;
                                                       Base_e e_role;
                                                       Base_e (Var { name = "cout"; index = 0 } |> mk_dsyntax "")])
      in

       let is_arr = is_array_typ t in
       
       (* bt is the base type and sl is the secret label *)
       let sl = l |> get_opt |> get_secret_label in

       (* list of array dimensions, if any *)
       let el = if is_arr then t |> get_array_bt_and_dimensions |> snd else [] in

       (* expression that we will put in the lhs for each output gate, and cout eventually *)
       let elmt_of_e =
         let aux (e:expr) :codegen_expr =
           Base_e (el |> List.fold_left (fun (i, e) _ ->
                             let i_var = { name = "i" ^ (string_of_int i); index = 0 } in
                             i + 1,
                             Array_read (e, Var i_var |> mk_dsyntax "") |> mk_dsyntax ""
                           ) (0, e) |> snd)
         in
         aux e
       in

       (* now we are going to put two nested loops, one for output gates, and one for cout *)
       let output_gate_loops =
         let aux (s:codegen_stmt) :codegen_stmt =
           List.fold_right (fun e (i, s) ->
               let i_var = { name = "i" ^ (string_of_int i); index = 0 } in
               i - 1,
               For_codegen (Base_e (Var i_var |> mk_dsyntax ""),
                            Base_e (Const (UInt32C (Uint32.of_int 0)) |> mk_dsyntax ""),
                            Base_e e,
                            s)
             ) el (List.length el - 1, s) |> snd
         in
         aux (App_codegen ("add_to_output_queue", [Base_e (Var { name = "out_q"; index = 0 } |> mk_dsyntax "");
                                                   Output_g (r, sl, elmt_of_e);
                                                   Base_e e_role;
                                                   Base_e (Var { name = "cout"; index = 0 } |> mk_dsyntax "")]))
       in

       o_codegen_stmt g (Seq_codegen (print_output_msg, output_gate_loops))

  | Skip s -> (if s = "" then o_null else seq o_newline (seq (o_comment s) o_newline)), g
           
  | _ -> failwith "codegen_stmt: impossible branch"

and read_or_write_interim (g:gamma) (write:bool) (e_var:expr) (t:typ) (f:string) :comp =
  let f = if not (Config.get_shares_dir () = "") then (Config.get_shares_dir () ^ "/" ^ f) else f in

  let bt, l = get_bt_and_label t in
  let is_arr = is_array_typ t in     

  (* bt is the base type and l is the label *)
  let l = l |> get_opt in
  (* list of array dimensions, if any *)
  let el = if is_arr then t |> get_array_bt_and_dimensions |> snd else [] in

  (* expression that we will put in the lhs for each output gate, and read/write eventually *)
  let elmt_of_e =
    let aux (e:expr) :codegen_expr =
      Base_e (el |> List.fold_left (fun (i, e) _ ->
                        let i_var = { name = "i" ^ (string_of_int i); index = 0 } in
                        i + 1,
                        Array_read (e, Var i_var |> mk_dsyntax "") |> mk_dsyntax ""
                      ) (0, e_var) |> snd)
    in
    aux e_var
  in
  
  let fstream_add_name = get_fresh_var () in
  let fstream_rand_name = get_fresh_var () in
  let decl_fstream_add =
    let comment =
      "File for " ^ (if write then "writing" else "reading") ^ " variable " ^ (e_var |> get_var |> (fun v -> v.name))
    in
    Seq_codegen (Base_s (Skip comment |> mk_dsyntax ""), Open_file (write, fstream_add_name, f ^ ".sh"))
  in
  let decl_fstream_rand = Open_file (write, fstream_rand_name, f ^ ".rand.sh") in
  
  (* now we are going to put nested loops *)
  let loops =
    let aux (s:codegen_stmt) :codegen_stmt =
      List.fold_right (fun e (i, s) ->
          let i_var = { name = "i" ^ (string_of_int i); index = 0 } in
          i - 1,
          For_codegen (Base_e (Var i_var |> mk_dsyntax ""),
                       Base_e (Const (UInt32C (Uint32.of_int 0)) |> mk_dsyntax ""),
                       Base_e e,
                       s)
        ) el (List.length el - 1, s) |> snd
    in
    let s_body =
      if l = Public then
        if write then Cout (fstream_add_name, elmt_of_e, bt)
        else Cin (fstream_add_name, elmt_of_e, bt)
      else
        let sl = get_secret_label l in
        let circ_var =
          (if sl = Arithmetic then Var {name = "acirc"; index = 0 }
           else Var {name = "ycirc"; index = 0 })
          |> mk_dsyntax ""
        in
        if write then
          App_codegen ("write_share", [elmt_of_e;
                                       Base_e circ_var;
                                       Base_e (Var {name = "bitlen"; index = 0 } |> mk_dsyntax "");
                                       Base_e (Var {name = "role"; index = 0 } |> mk_dsyntax "");
                                       Base_e (Var {name = fstream_add_name; index = 0 } |> mk_dsyntax "");
                                       Base_e (Var {name = fstream_rand_name; index = 0 } |> mk_dsyntax "");
                                       Base_e (Var { name = "out_q"; index = 0 } |> mk_dsyntax "")])
        else
          let fname =
            if Config.get_bitlen () = 32 then "read_share<uint32_t>"
            else "read_share<uint64_t>"
          in
          Assign_codegen (elmt_of_e,
                          App_codegen_expr (fname,
                                            [Base_e circ_var;
                                             Base_e (Var {name = "role"; index = 0 } |> mk_dsyntax "");
                                             Base_e (Var {name = "bitlen"; index = 0 } |> mk_dsyntax "");
                                             Base_e (Var {name = fstream_add_name; index = 0 } |> mk_dsyntax "");
                                             Base_e (Var {name = fstream_rand_name; index = 0 } |> mk_dsyntax "")]))
    in
    aux s_body
  in

  let s = Seq_codegen (decl_fstream_add, Seq_codegen (decl_fstream_rand, loops)) in
  let s = if write then s
          else Seq_codegen (s, Seq_codegen (Close_file fstream_rand_name, Close_file fstream_add_name))
  in

  o_codegen_stmt g s |> fst

and o_codegen_stmt (g:gamma) (s:codegen_stmt) :comp * gamma =
  match s with
  | Base_s s -> o_stmt g s

  | App_codegen (f, args) -> o_with_semicolon (o_app (o_str f) (List.map (o_codegen_expr g) args)), g

  | Cin (s, x, _) ->
     (if Config.get_dummy_inputs () then
        o_with_semicolon (seq (o_codegen_expr g x) (o_str (" = rand()")))
      else
        o_with_semicolon (seq (o_str s) (seq (o_str " >> ") (o_codegen_expr g x)))),
     g

  | Cout (s, e, _) ->
     o_with_semicolon (seq (o_str s) (seq (o_str " << ") (seq (o_paren (o_codegen_expr g e)) (o_str " << endl")))),
     g

  | Dump_interim (e_var, t, f) when is_var e_var -> read_or_write_interim g true e_var t f, g

  | Read_interim (e_var, t, f) when is_var e_var -> read_or_write_interim g false e_var t f, g

  | Open_file (write, x, f) ->
     (if write then
        let c = o_str ("ofstream " ^ x ^ ";\n" ^ x ^ ".open(\"" ^ f ^ "\", ios::out | ios::trunc);\n") in
        out_files := !out_files @ [x];
        c
      else o_str ("ifstream " ^ x ^ ";\n" ^ x ^ ".open(\"" ^ f ^ "\", ios::in);\n")),
     g

  | Close_file f -> o_str (f ^ ".close ();\n"), g

  | Dump_interim _ -> failwith "Impossible! we don't support dumping of arbitrary shares"

  | Read_interim _ -> failwith "Impossible! we don't support reading of arbitrary shares"

  | Assign_codegen (e1, e2) -> o_with_semicolon (o_assign (o_codegen_expr g e1) (o_codegen_expr g e2)), g
           
  | For_codegen (e1, e2, e3, s) ->
     let i_var =
       match e1 with
       | Base_e e1 -> get_var e1
       | _ -> failwith "Codegen::o_codegen_stmt: expected for loop index to be a Base_e"
     in
     let g_body = [i_var,
                   Base (Int32, Some Public) |> mk_dsyntax ""] |> push_local_scope g
     in
     o_for (o_codegen_expr g e1) (o_codegen_expr g e2) (o_codegen_expr g e3) (o_codegen_stmt g_body s |> fst),
     g

  | If_codegen (_, e, s_then, s_else_opt) ->
     let g_body = [] |> push_local_scope g in
     o_ite (o_codegen_expr g e) (o_codegen_stmt g_body s_then |> fst)
           (map_opt s_else_opt (fun s -> s |> o_codegen_stmt g_body |> fst)),
     g

  | Seq_codegen (s1, s2) ->
     let c1, g = o_codegen_stmt g s1 in
     let c2, g = o_codegen_stmt g s2 in
     seq c1 (seq o_newline c2), g

  | _ -> failwith "o_codegen_stmt: impossible case"

let o_binder (b:binder) :comp =
  let o_typ (t:typ) :comp = if is_array_typ t then seq (o_typ t) (o_str "&") else o_typ t in
  seq (b |> snd |> o_typ) (seq o_space (b |> fst |> o_var))

let o_global (g0:gamma) (d:global) :comp * gamma =
  match d.data with
  | Fun (_, fname, bs, body, ret_t) ->
     let g = enter_fun g0 d.data in
     let header = seq (o_ret_typ ret_t) (seq o_space (o_hd_and_args (o_str fname) (List.map o_binder bs))) in
     seq header (seq (o_str "{\n") (seq (o_stmt g body |> fst) (o_str "\n}\n"))),
     add_fun g0 d.data
  | Extern_fun (_, fname, bs, ret_t) ->
     let header = seq (o_ret_typ ret_t) (seq o_space (o_hd_and_args (o_str fname) (List.map o_binder bs))) in
     o_with_semicolon (seq (o_str "extern ") header),
     add_fun g0 d.data
  | Global_const (t, e_var, init) ->
     seq (o_with_semicolon (seq (o_str "const ") (o_decl (o_typ t) (o_expr g0 e_var) (Some (o_expr g0 init))))) o_newline,
     add_global_const g0 d.data

let prelude_string :string =
"\
/*\n\
This is an autogenerated file, generated using the EzPC compiler.\n\
*/\n\
#include<vector>\n\
#include<math.h>\n\
#include<cstdlib>\n\
#include<iostream>\n\
#include<fstream>\n\
\n\
using namespace std;\n\
\n\
uint32_t public_lrshift(uint32_t x, uint32_t y){\n\
  return (x >> y);\n\
}\n\
\n\
int32_t public_lrshift(int32_t x, uint32_t y){\n\
  return ((int32_t)(((uint32_t)x) >> y));\n\
}\n\
\n\
uint64_t public_lrshift(uint64_t x, uint64_t y){\n\
  return (x >> y);\n\
}\n\
\n\
int64_t public_lrshift(int64_t x, uint64_t y){\n\
  return ((int64_t)(((uint64_t)x) >> y));\n\
}\n\
\n\
template<typename T>\n\
vector<T> make_vector(size_t size) {\n\
  return std::vector<T>(size);\n\
}\n\
\n\
template <typename T, typename... Args>\n\
  auto make_vector(size_t first, Args... sizes)\n\
{\n\
  auto inner = make_vector<T>(sizes...);\n\
  return vector<decltype(inner)>(first, inner);\n\
}\n\
\n\
template<typename T>\n\
ostream& operator<< (ostream &os, const vector<T> &v)\n\
{\n\
  for(auto it = v.begin (); it != v.end (); ++it) {\n\
    os << *it << endl;\n\
  }\n\
  return os;\n\
}\n\
\n\
"
                                   
let aby_prelude_string (bitlen:int) :string =
"\
#include \"ezpc.h\"\n\
ABYParty *party;\n\
Circuit* ycirc;\n\
Circuit* acirc;\n\
Circuit* bcirc;\n\
uint32_t bitlen = " ^ string_of_int bitlen ^ ";\n\
output_queue out_q;\n\
e_role role;\n"

let aby_main_prelude_string :string =
"role = role_param;\n\
party = new ABYParty(role_param, address, port, seclvl, bitlen, nthreads, mt_alg, 520000000);\n\
std::vector<Sharing*>& sharings = party->GetSharings();\n\
ycirc = (sharings)[S_YAO]->GetCircuitBuildRoutine();\n\
acirc = (sharings)[S_ARITH]->GetCircuitBuildRoutine();\n\
bcirc = (sharings)[S_BOOL]->GetCircuitBuildRoutine();\n"

let o_one_program ((globals, main):global list * codegen_stmt) (ofname:string) :unit =
  let prelude =
    if Config.get_codegen () = Config.CPP then o_str prelude_string
    else
      o_str (prelude_string ^ "\n" ^ (aby_prelude_string (Config.get_bitlen ())))
  in

  let main_header =
    if Config.get_codegen () = Config.CPP then o_str "\n\nint main () {\n"
    else o_str "\n\nint64_t ezpc_main (e_role role_param, char* address, uint16_t port, seclvl seclvl,\n\
                uint32_t nvals, uint32_t nthreads, e_mt_gen_alg mt_alg,\n\
                e_sharing sharing) {\n"
  in

  let main_prelude =
    if Config.get_codegen () = Config.CPP then o_null
    else seq (o_str aby_main_prelude_string) o_newline
  in
  
  let main_prelude, g =
    let c_globals, g = List.fold_left (fun (c, g) d ->
                           let c_global, g = o_global g d in
                           seq c (seq o_newline c_global), g) (o_null, empty_env) globals
    in
    seq prelude (seq c_globals (seq main_header main_prelude)), g
  in
  let main_body, g = o_codegen_stmt ([] |> push_local_scope g) main in
  let main_end =
    if Config.get_codegen () = Config.ABY then
      o_str "\nparty->ExecCircuit();\nflush_output_queue(out_q, role, bitlen);"
    else o_null
  in

  let out_files_close_stmt =
    !out_files |> List.fold_left (fun s f -> Seq_codegen (s, Close_file f)) (Base_s (Skip "" |> mk_dsyntax ""))
  in

  let file = seq (seq main_prelude (seq main_body main_end)) (o_codegen_stmt g out_files_close_stmt |> fst) in
  let file = seq file (o_str "\nreturn 0;\n}\n") in
  
  let b = Buffer.create 0 in
  file b;
  let fd = open_out ofname in
  Buffer.contents b |> fprintf fd "%s\n";

  close_out fd

let o_program ((globals, mains):codegen_program) (ofname_prefix:string) :unit =
  mains |> List.fold_left (fun i m ->
               o_one_program (globals, m) (ofname_prefix ^ (string_of_int i) ^ ".cpp");
               out_files := [];
               i + 1) 0 |> ignore
