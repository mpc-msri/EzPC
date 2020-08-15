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

let o_hd_and_args (head:comp) (args:comp list) :comp =
  match args with
  | []     -> seq head (seq (o_str "(") (o_str ")"))
  | hd::tl ->
     let c = seq head (seq (o_str "(") hd) in
     let c = List.fold_left (fun c arg -> seq c (seq (o_str ", ") arg)) c tl in
     seq c (o_str ")")

let o_app (head:comp) (args:comp list) :comp = o_hd_and_args head args

let o_cbfunction (sl:secret_label) (f:comp) (args:comp list) :comp = 
  o_app f args

let o_sunop (l:secret_label) (op:unop) (c:comp) :comp =
  let err_unsupp (s:string) = failwith ("Codegen: Operator: " ^ s ^ " is not supported by this backend.") in
  match op with
  | U_minus -> failwith "Codegen: unary minus is not being produced by lexer or parser right now."
  | Bitwise_neg -> err_unsupp "Bitwise_neg"
  | Not -> err_unsupp "Boolean_not"
  
let o_sbinop (l:secret_label) (op:binop) (c1:comp) (c2:comp) :comp =
  let err (s:string) = failwith ("Codegen: Operator: " ^ s ^ " should have been desugared") in
  let err_unsupp (s:string) = failwith ("Codegen: Operator: " ^ s ^ " is not supported by this backend.") in
  let infix_op (op_comp :comp) = seq c1 (seq op_comp c2) in
  match op with
  | Sum                -> o_cbfunction l (o_str "SecretAdd") [c1; c2]
  | Sub                -> o_cbfunction l (o_str "SecretSub") [c1; c2]
  | Mul                -> o_cbfunction l (o_str "SecretMult") [c1; c2]
  | Greater_than       -> err_unsupp "Greater_than"
  | Div                -> err_unsupp "Div"
  | Mod                -> err_unsupp "MOD"
  | Less_than          -> err "LT"
  | Is_equal           -> err "EQ"
  | Greater_than_equal -> err "GEQ"
  | Less_than_equal    -> err "LEQ"
  | R_shift_a          -> err_unsupp "Arithmetic_right_shift"
  | L_shift            -> infix_op (o_str "<<")
  | Bitwise_and        -> err_unsupp "Bitwise_and"
  | Bitwise_or         -> err_unsupp "Bitwise_or"
  | Bitwise_xor        -> err_unsupp "Bitwise_xor"
  | And                -> err_unsupp "Boolean_and"
  | Or                 -> err_unsupp "Boolean_or"
  | Xor                -> err_unsupp "Boolean_xor"
  | R_shift_l          -> err_unsupp "Logical_right_shift"
  | Pow                -> err_unsupp "Pow"
               
let o_pconditional (c1:comp) (c2:comp) (c3:comp) :comp =
  seq c1 (seq (o_str " ? ") (seq c2 (seq (o_str " : ") c3)))

let o_subsumption (src:label) (tgt:secret_label) (t:typ) (arg:comp) :comp =
  match src with
    | Public -> o_app (o_str "funcSSCons") [arg]
    | Secret Arithmetic
    | Secret Boolean -> failwith "Codegen: Subsumption from secrets is not allowed for this backend."

let o_basetyp (t:base_type) :comp =
  let uint32_basetype_str :string = if Config.get_porthos2pc_backend () = OT then "uint32_t" else "uint64_t" in
  let int32_basetype_str :string = if Config.get_porthos2pc_backend () = OT then "int32_t" else "int64_t" in
  match t with
  | UInt32 -> o_str uint32_basetype_str
  | UInt64 -> o_str "uint64_t"
  | Int32  -> o_str int32_basetype_str
  | Int64  -> o_str "int64_t"
  | Bool   -> o_str uint32_basetype_str

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
      | Is_equal -> fn_name "equals"
      | Less_than -> fn_name "lt"
      | Greater_than_equal -> fn_name "geq"
      | Less_than_equal -> fn_name "leq"
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
  let uint32_basetype_str :string = if Config.get_porthos2pc_backend () = OT then "uint32_t" else "uint64_t" in
  let rec o_array_read_rec (ga:gamma) (ea:expr) : (comp*comp*int) =
    match ea.data with
    | Array_read (e1,e2) -> 
        (let tt = get_opt (typeof_expr ga e1) in
        match tt.data with 
        | Base (_,_) -> failwith "Unknown control flow"
        | Array (_,_,e2size) -> 
            (let (e1size,e1idx,iter) = o_array_read_rec ga e1 in  
            let e2idx = o_expr e2 in
            ((seq e1size (seq (o_str ",") (o_expr e2size))),(seq e1idx (seq (o_str ",") e2idx)),iter+1)))
    | _ -> (o_expr ea, o_str "", 0)
  in
  match e.data with
  | Role r -> o_role r

  | Const (Int32C n) -> seq (o_str (" (int32_t)")) (o_int32 n)

  | Const (Int64C n) -> seq (o_str (" (int64_t)")) (o_int64 n)

  | Const (UInt32C n) -> seq (o_str (" (" ^ uint32_basetype_str ^ ")")) (o_uint32 n)

  | Const (UInt64C n) -> seq (o_str (" (uint64_t)")) (o_uint64 n)

  | Const (BoolC b) -> o_bool b
    
  | Var s -> o_var s

  | Unop (op, e, Some Public) -> seq (o_punop op) (seq o_space (o_expr e))
  
  | Unop (op, e, Some (Secret s)) -> seq (o_sunop s op (o_expr e)) (seq o_space (o_expr e))

  | Binop (op, e1, e2, Some Public) ->
     o_paren (match op with
              | R_shift_l -> o_app (o_str "public_lrshift") [o_expr e1; o_expr e2]
              | Pow -> o_app (o_str "pow") [o_expr e1; o_expr e2]
              | _ -> seq (o_expr e1) (seq o_space (seq (o_pbinop op) (seq o_space (o_expr e2)))))

  | Binop (op, e1, e2, Some (Secret s)) -> o_secret_binop g op s e1 e2

  | Conditional (e1, e2, e3, lopt) -> o_codegen_expr (Conditional_codegen (Base_e e1, Base_e e2, Base_e e3, (get_opt lopt)))

  | Array_read (e1, e2) -> 
      let (esize,eidx,ndarr) = o_array_read_rec g e in 
      seq (o_str ("Arr" ^ (string_of_int ndarr) ^ "DIdxRowM(")) (seq esize (seq eidx (o_str ")")))

  | App (f, args) -> o_app (o_str f) (List.map o_expr args)

  | Subsumption (e, l1, Secret l2) -> o_subsumption l1 l2 (typeof_expr g e |> get_opt) (o_expr e)

  | _ -> failwith "o_expr: impossible branch"

and o_codegen_expr (g:gamma) (e:codegen_expr) :comp =
  let o_expr = o_expr g in
  let o_codegen_expr = o_codegen_expr g in
  match e with
  | Base_e e -> o_expr e
              
  | Input_g (r, sl, s, bt) -> o_str s.name

  | Dummy_g (sl, bt) -> o_str "0"
    
  | Output_g (r, sl, e) -> 
      let o_reveal_bitmask = 
        match r with 
        | Client -> o_str "2" (*Bitmask = 10*)
        | Server -> o_str "1" (*Bitmask = 01*)
        | Both -> o_str "3" (*Bitmask = 11*)
      in
      o_cbfunction sl (o_str "funcReconstruct2PCCons") [o_codegen_expr e; o_reveal_bitmask]

  | Conditional_codegen (e1, e2, e3, l) ->
     let c1, c2, c3 = o_codegen_expr e1, o_codegen_expr e2, o_codegen_expr e3 in
     (match l with
      | Public -> o_pconditional c1 c2 c3
      | Secret sl -> failwith "Secret conditionals not allowed for this backend.")

  | App_codegen_expr (f, el) -> o_app (o_str f) (List.map o_codegen_expr el)
  | Clear_val _ -> failwith ("Codegen_expr Clear_val is unsupported by this backend.") 

let rec o_typ_rec (t:typ) :comp =
  let uint32_basetype_str :string = if Config.get_porthos2pc_backend () = OT then "uint32_t" else "uint64_t" in
  match t.data with
  | Base (Int64, Some (Secret _)) -> o_str "uint64_t"
  | Base (Int32, Some (Secret _)) -> o_str uint32_basetype_str
  | Base (_, Some (Secret _)) -> failwith "Codegen: For secret shared variables, only int64 is allowed by this backend."
  | Base (bt, _) -> o_basetyp bt
  | Array (quals,tt, _) -> seq (if quals |> List.mem Immutable then o_str "const " else o_null) (o_typ_rec tt) 
                                      
let o_typ (t:typ) :comp =
  match t.data with
  | Array (_,_, _) -> seq (o_typ_rec t) (o_str "*")
  | _ -> o_typ_rec t

let o_ret_typ (t:ret_typ) :comp =
  match t with
  | Typ t -> o_typ t
  | Void _ -> o_str "void"
             
let o_array_init (g:gamma) (t:typ) :comp =
  let t, l = get_array_bt_and_dimensions t in
  let s = seq (o_str "make_array<") (seq (o_typ t) (o_str ">")) in
  o_app s (List.map (o_expr g) l)

let o_for (index:comp) (lower:comp) (upper:comp) (body:comp) :comp =
  let uint32_basetype_str :string = if Config.get_porthos2pc_backend () = OT then "uint32_t" else "uint64_t" in
  let init = seq (o_str ("for (" ^ uint32_basetype_str ^ " ")) (seq index (seq (o_str " = ") lower)) in
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

     (* this is the innermost loop body *)
     let base_init =
       let role_var = { name = "party"; index = 0 } in
       let r_cmp = Base_e (Binop (Is_equal, Var role_var |> mk_dsyntax "", Role r |> mk_dsyntax "", Some Public) |> mk_dsyntax "") in
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

     (* stitch *)
     o_codegen_stmt g (Seq_codegen (decl, Seq_codegen (decl_tmp, loops)))

  | Output (e_role, e, Some t) when is_role e_role ->
     let r = get_role e_role in
     let bt, l = get_bt_and_label t in
     if not (l |> get_opt |> is_secret_label) then o_codegen_stmt g (Cout ("cout", Base_e e, bt))
     else
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
         aux (Cout ("cout", (Output_g (r, sl, elmt_of_e)), bt))
       in

       o_codegen_stmt g output_gate_loops

  | Skip s -> (if s = "" then o_null else seq o_newline (seq (o_comment s) o_newline)), g
           
  | _ -> failwith "codegen_stmt: impossible branch"

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

  | Dump_interim _ | Read_interim _ -> failwith "Codegen: Dumping/Reading of interim shares is not supported by this backend."

  | Open_file _ | Close_file _ -> failwith "Codegen: Opening/Closing of file is not supported by this backend."

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
  let o_typ (t:typ) :comp = o_typ t in
  seq (b |> snd |> o_typ) (seq o_space (b |> fst |> o_var))

let o_global (g0:gamma) (d:global) :comp * gamma =
  match d.data with
  | Fun (_, fname, bs, body, ret_t) ->
     let g = enter_fun g0 d.data in
     let header = seq (o_ret_typ ret_t) (seq o_space (o_hd_and_args (o_str fname) (List.map o_binder bs))) in
     seq header (seq (o_str "{\n") (seq (o_stmt g body |> fst) (o_str "\n}\n"))),
     add_fun g0 d.data
  | Extern_fun (_, fname, bs, ret_t) ->
     o_null,
     add_fun g0 d.data
  | Global_const (t, e_var, init) ->
     seq (o_with_semicolon (seq (o_str "const ") (o_decl (o_typ t) (o_expr g0 e_var) (Some (o_expr g0 init))))) o_newline,
     add_global_const g0 d.data

let prelude_string :string =
"\
/*\n\
This is an autogenerated file, generated using the EzPC compiler.\n\
*/\n\
"
                                   
let porthos_prelude_string (hash_define_str:string) :string =
  let bitlen = Config.get_actual_bitlen () |> string_of_int in
"\n\
#define BITLEN_"
^
bitlen
^
"\n\
#define "
^
hash_define_str
^
"
\n\

#include \"globals.h\"\n\
#include \"functionalities_wrapper.h\"\n\
#include <iostream>\n\
using namespace std;\n\
\n\
template<typename T>\n\
T* make_array(size_t s1){\n\
  return new T[s1];\n\
}\n\
\n\
template<typename T>\n\
T* make_array(size_t s1, size_t s2){\n\
  return new T[s1*s2];\n\
}\n\
\n\
template<typename T>\n\
T* make_array(size_t s1, size_t s2, size_t s3){\n\
  return new T[s1*s2*s3];\n\
}\n\
\n\
template<typename T>\n\
T* make_array(size_t s1, size_t s2, size_t s3, size_t s4){\n\
  return new T[s1*s2*s3*s4];\n\
}\n\
\n\
template<typename T>\n\
T* make_array(size_t s1, size_t s2, size_t s3, size_t s4, size_t s5){\n\
  return new T[s1*s2*s3*s4*s5];\n\
}\n\
"

let porthos_main_decl :string =
"
\n\
int main(int argc, char** argv)\n\
{\n\
"

let porthos_main_prelude_string :string =
"\
ArgMapping amap;\n\
int port = 32000;\n\
string serverAddr = \"127.0.0.1\";\n\
\n\
amap.arg(\"r\", party, \"Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2\");\n\
amap.arg(\"p\", port, \"Port Number\");\n\
amap.arg(\"ip\", serverAddr, \"IP Address of server (ALICE)\");\n\
amap.parse(argc, argv);\n\
\n\
assert(party==sci::ALICE || party==sci::BOB);\n\
\n\
checkIfUsingEigen();\n\
for(int i=0;i<numThreads;i++){\n\
ioArr[i] = new sci::NetIO(party==sci::ALICE ? nullptr : serverAddr.c_str(), port+i);\n\
otInstanceArr[i] = new sci::IKNP<sci::NetIO>(ioArr[i]);\n\
prgInstanceArr[i] = new sci::PRG128();\n\
kkotInstanceArr[i] = new sci::KKOT < sci::NetIO > (ioArr[i]);\n\
matmulInstanceArr[i] = new Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>(party, bitlength, ioArr[i], otInstanceArr[i], nullptr);\n\
if (i == 0) {\n\
otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], party, baseForRelu, bitlength);\n\
} \n\
else if (i == 1) {\n\
otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], 3-party, baseForRelu, bitlength);\n\
} \n\
else if (i & 1) {\n\
otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], 3-party, baseForRelu, bitlength, false);\n\
otpackArr[i]->copy(otpackArr[1]);\n\
} \n\
else {\n\
otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], party, baseForRelu, bitlength, false);\n\
otpackArr[i]->copy(otpackArr[0]);\n\
}\n\
}\n\
\n\
io = ioArr[0];\n\
iknpOT = new sci::IKNP<sci::NetIO>(io);\n\
iknpOTRoleReversed = new sci::IKNP<sci::NetIO>(io); //TCP is full duplex -- so both side OT on same TCP should be good\n\
kkot = new sci::KKOT<sci::NetIO>(io);\n\
prg128Instance = new sci::PRG128();\n\
otpack = new sci::OTPack<sci::NetIO>(io, party, baseForRelu, bitlength);\n\
\n\
matmulImpl = new Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>(party, bitlength, io, iknpOT, iknpOTRoleReversed);\n\
\n\
\n\
#ifdef SCI_OT\n\
reluImpl = new ReLURingProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,otpack);\n\
maxpoolImpl = new MaxPoolProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,0,otpack,reluImpl);\n\
argmaxImpl = new ArgMaxProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,0,otpack,reluImpl);\n\
#endif\n\
\n\
#ifdef SCI_HE\n\
reluImpl = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack);\n\
maxpoolImpl = new MaxPoolProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack,reluImpl);\n\
argmaxImpl = new ArgMaxProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack,reluImpl);\n\
heConvImpl = new ConvField(party,io);\n\
heFCImpl = new FCField(party,io);\n\
heProdImpl = new ElemWiseProdField(party, io);\n\
assertFieldRun();\n\
#endif\n\
#ifdef MULTITHREADED_NONLIN\n\
#ifdef SCI_OT\n\
for(int i = 0; i < numThreads; i++) {\n\
if (i & 1) {\n\
reluImplArr[i] = new ReLURingProtocol<sci::NetIO, intType>(3-party,RING,ioArr[i],bitlength,baseForRelu,otpackArr[i]);\n\
maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(3-party,RING,ioArr[i],bitlength,baseForRelu,0,otpackArr[i],reluImplArr[i]);\n\
} \n\
else {\n\
reluImplArr[i] = new ReLURingProtocol<sci::NetIO, intType>(party,RING,ioArr[i],bitlength,baseForRelu,otpackArr[i]);\n\
maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(party,RING,ioArr[i],bitlength,baseForRelu,0,otpackArr[i],reluImplArr[i]);\n\
}\n\
}\n\
#endif\n\
#ifdef SCI_HE\n\
for(int i = 0; i < numThreads; i++) {\n\
if (i & 1) {\n\
reluImplArr[i] = new ReLUFieldProtocol<sci::NetIO, intType>(3-party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i]);\n\
maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(3-party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i],reluImplArr[i]);\n\
} \n\
else {\n\
reluImplArr[i] = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i]);\n\
maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i],reluImplArr[i]);\n\
}\n\
}\n\
#endif\n\
#endif\n\
\n\
if (party==sci::ALICE){\n\
iknpOT->setup_send();\n\
iknpOTRoleReversed->setup_recv();\n\
}\n\
else if (party==sci::BOB){\n\
iknpOT->setup_recv();\n\
iknpOTRoleReversed->setup_send();\n\
}\n\
cout<<\"After base ots, communication = \"<<(io->counter)<<\" bytes\"<<endl;
"

let o_one_program ((globals, main):global list * codegen_stmt) (ofname:string) :unit =
  let (hash_define_str, main_prelude) = 
    let modulo_str = Config.get_modulo () |> Uint64.to_string in
    let hash_define_str = 
      if (Config.get_porthos2pc_backend () = OT) then "SCI_OT"
      else "SCI_HE"
    in
    let main_prelude = 
      if (Config.get_porthos2pc_backend () = OT) then begin
        (* OT case *)
        if Config.get_modulo () = Uint64.shift_left (Uint64.of_int 1) (Config.get_actual_bitlen ()) then ""
        else failwith "Modulo can only be (1<<bitlen) for OT case for Porthos2PC backend."
      end
      else begin
        (* HE case *)
        if Config.get_modulo () <> Uint64.shift_left (Uint64.of_int 1) (Config.get_actual_bitlen ()) then 
          "prime_mod = " ^ modulo_str ^ ";\n\
          "
        else ""
      end
    in 
    (hash_define_str, main_prelude) 
  in

  let prelude = o_str (prelude_string ^ "\n" ^ (porthos_prelude_string hash_define_str))
  in

  let main_header = o_str porthos_main_decl
  in

  let main_prelude = seq (seq (o_str main_prelude) (o_str porthos_main_prelude_string)) o_newline
  in
  
  let main_prelude, g =
    let c_globals, g = List.fold_left (fun (c, g) d ->
                           let c_global, g = o_global g d in
                           seq c (seq o_newline c_global), g) (o_null, empty_env) globals
    in
    seq prelude (seq c_globals (seq main_header main_prelude)), g
  in
  let main_body, g = o_codegen_stmt ([] |> push_local_scope g) main in
  let main_end = o_null
  in

  let file = seq main_prelude (seq main_body main_end) in
  let file = seq file (o_str "\n}\n") in
  
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
