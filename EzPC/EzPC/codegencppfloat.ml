(*

Authors: Anwesh Bhattacharya.

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
open String
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

let o_sunop (l:secret_label) (op:unop) (c:comp) :comp =
  failwith "Codegen: unary minus is not being produced by lexer or parser right now."   
               
let o_pconditional (c1:comp) (c2:comp) (c3:comp) :comp =
  seq c1 (seq (o_str " ? ") (seq c2 (seq (o_str " : ") c3)))
  
let o_sconditional (l:secret_label) (opstr:string) (c_cond:comp) (c_then:comp) (c_else:comp) :comp =
  o_app (opstr ^ "->if_else" |> o_str) [c_cond; c_then; c_else] 

let o_subsumption (src:label) (tgt:secret_label) (t:typ) (arg:comp) :comp =
  match src with
  | Public ->
    (match t.data with
    | Base (UInt32, _) -> o_app (o_str "__public_int_to_arithmetic") [arg; o_str "false"; o_str "32"] 
    | Base (UInt64, _) -> o_app (o_str "__public_int_to_arithmetic") [arg; o_str "false"; o_str "64"]
    | Base (Int32, _) -> o_app (o_str "__public_int_to_arithmetic") [arg; o_str "true"; o_str "32"]
    | Base (Int64, _) -> o_app (o_str "__public_int_to_arithmetic") [arg; o_str "true"; o_str "64"]
    | Base (Float, _) -> o_app (o_str "__public_float_to_baba") [arg]
    | Base (Bool, _) -> o_app (o_str "__public_bool_to_boolean") [arg]
    | _ -> failwith "Impossible source type for subsumption")
  | _ -> failwith "o_subsumption : Expected impossible branch. Should have been handled by infer.ml"

let o_sectyp (t:base_type) :comp =
  let str =
    match t with
    | UInt32 | UInt64 | Int32 | Int64 -> "FixArray"
    | Float  -> "FPArray"
    | Bool   -> "BoolArray"
  in o_str str

let basetyp_to_sectyp (t:base_type) : string =
  match t with
  | UInt32 | UInt64 | Int32 | Int64 -> "uint64_t"
  | Float -> "float"
  | Bool -> "uint8_t"

let basetyp_to_cpptyp (t:base_type) : string =
  match t with
    | UInt32 -> "uint32_t"
    | UInt64 -> "uint64_t"
    | Int32 -> "int32_t"
    | Int64 -> "int64_t"
    | Float -> "float"
    | Bool -> "bool"
 
let o_basetyp (t:base_type) :comp = t |> basetyp_to_cpptyp |> o_str

let basetype_to_secfloat (t:base_type) :string =
  match t with
  | UInt32 | UInt64 | Int32 | Int64 -> "fix"
  | Float -> "fp"
  | Bool -> "bool"

let basetype_to_secfloat_backend (t:base_type) :string = "__" ^ (basetype_to_secfloat t) ^ "_op"

let basetype_to_secfloat_pub (t:base_type) :string = "__" ^ (basetype_to_secfloat t) ^ "_pub"

let rec o_secret_binop (g:gamma) (op:binop) (sl:secret_label) (e1:expr) (e2:expr) :comp =
  let backend = e1 |> typeof_expr g |> get_opt |> get_bt_and_label |> fst |> basetype_to_secfloat_backend in
  let fn_name =
    match op with
    | Sum -> "add"
    | Sub -> "sub"
    | Mul -> "mul"
    | Div -> "div"
    | Less_than -> "LT"
    | Less_than_equal -> "LE"
    | Greater_than -> "GT"
    | Greater_than_equal -> "GE"
    | And -> "AND"
    | Or -> "Or"
    | _ -> failwith "o_secret_binop : This binop hasn't been implemented yet"
  in
  let app = App (backend ^ "->" ^ fn_name, [e1; e2]) in
  o_expr g (app |> mk_dsyntax "")

and o_expr (g:gamma) (e:expr) :comp =
  let o_expr = o_expr g in
  let o_codegen_expr = o_codegen_expr g in
  match e.data with
  | Role r -> o_role r
  | Const (Int32C n) -> o_int32 n
  | Const (Int64C n) -> o_int64 n
  | Const (UInt32C n) -> o_uint32 n
  | Const (UInt64C n) -> o_uint64 n
  | Const (FloatC f) -> o_float f 
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
  match e with
  | Codegen_String s -> o_str s

  | Base_e e -> o_expr e 
              
  | Clear_val (e, bt) -> seq (o_codegen_expr e) (seq (o_str "->get_clear_value<") (seq (o_basetyp bt) (o_str ">()")))

  | Conditional_codegen (e1, e2, e3, l) ->
     let c1, c2, c3 = o_codegen_expr e1, o_codegen_expr e2, o_codegen_expr e3 in
     (match l with
      | Public -> o_pconditional c1 c2 c3
      | Secret sl -> 
        let t = e2 |> get_base_e |> typeof_expr g |> get_opt |> get_bt_and_label |> fst in
        let opstr =
          match t with
          | Float -> "__fp_op"
          | Bool -> "__bool_op"
          | _ -> "__fix_op"
        in
        o_sconditional sl opstr c1 c2 c3
      )

  | App_codegen_expr (f, el) -> o_app (o_str f) (List.map o_codegen_expr el)

  | _  -> failwith "Invalid codegen_expr for this backend"

                                     
let o_typ (t:typ) :comp =  
  match t.data with
  | Base (bt, Some Public) | Base (bt, None) -> o_basetyp bt
  | Base (bt, Some (Secret _)) -> o_sectyp bt
  | Array (quals, _, _) -> seq (if quals |> List.mem Immutable then o_str "const " else o_null) (o_str "auto")

let o_ret_typ (t:ret_typ) :comp =
  match t with
  | Typ t -> o_typ t
  | Void _ -> o_str "void"
             
(*!! Need to ensure that array size arguments are passed appropriately !!*)
let o_array_init (g:gamma) (t:typ) :comp =
  let dims = get_array_bt_and_dimensions t |> snd in
  let t, lopt = get_bt_and_label t in
  let l = lopt |> get_opt in
  let o_dimargs = List.map (o_expr g) dims in
  if is_secret_label l then
    let party = "ALICE" in
    let o_args = [party] @
      (match t with
      | UInt32 -> ["false"; "32"]
      | UInt64 -> ["false"; "32"]
      | Int32 -> ["true"; "32"]
      | Int64 -> ["true"; "64"]
      | _ -> []) |> List.map o_str in
    let args = o_args @ o_dimargs in
    let make_typ =
      match t with
      | UInt32 | UInt64 | Int32 | Int64 -> "integer"
      | Float -> "float"
      | Bool -> "bool"
    in let s = ("make_vector_" ^ make_typ) |> o_str in
    o_app s args 
  else
    let make_typ = t |> basetyp_to_cpptyp in
    let s = ("make_vector<" ^ make_typ ^ ">") |> o_str in
    o_app s o_dimargs

let o_for (index:comp) (lower:comp) (upper:comp) (body:comp) :comp =
  let init = seq (o_str "for (uint32_t ") (seq index (seq (o_str " = ") lower)) in
  let term = seq index (seq (o_str " < ") upper) in
  let incr = seq index (o_str "++)") in
  let header = seq init (seq (o_str "; ") (seq term (seq (o_str "; ") (seq incr (o_str "{"))))) in
  seq header (seq o_newline (seq body (seq o_newline (o_str "}"))))

let o_while (guard:comp) (body:comp) :comp =
  seq (o_str "while (") (seq guard (seq (o_str ") {\n") (seq body (o_str "\n}\n"))))
  
let o_ite (c_if:comp) (c_then:comp) (c_else_opt:comp option) :comp =
  (* let c_if_then = seq (o_str "if (") (seq c_if (seq (o_str ") {") (seq o_newline (seq c_then (seq o_newline (o_str "}")))))) in *)
  (* Too many brackets, simplified expression below *)
  let c_if_then = seql [o_str "if ("; c_if; o_str ") {"; o_newline; 
        c_then; o_newline; 
        o_str "}"] in
  if is_none c_else_opt then c_if_then
  else (* seq c_if_then (seq (o_str " else {") (seq o_newline (seq (get_opt c_else_opt) (seq o_newline (o_str "}"))))) *)
       (* Too many brackets, simplified expression below *)
      seql [c_if_then; o_str " else {"; o_newline; 
            get_opt c_else_opt; o_newline; 
        o_str "}"]

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
      let alb = add_local_binding g (get_var e) t in
      if is_some init_opt then
        seql [o_typ t; o_space; o_expr g e; o_str " = "; init_opt |> get_opt |> o_expr g] |> s_smln, alb
      else
        if is_array_typ t then
          seql [o_typ t; o_space; o_expr g e; o_str " = "; o_array_init g t] |> s_smln, alb
        else 
          let bt, lab_opt = get_bt_and_label t in
          let lab = lab_opt |> get_opt in
          if is_secret_label lab then 
            let o_party = "ALICE" |> o_str in
            let o_sz = "1" |> o_str in
            let init_args = (fun f ->
              match bt with
              | UInt32 | Int32 | UInt64 | Int64 ->
                let o_sign =
                  (match bt with
                  | UInt32 | UInt64 -> o_str "false"
                  | Int32 | Int64 -> o_str "true"
                  | _ -> failwith "o_stmt : This was supposed to be an exhaustive match"
                  ) in
                let o_ell = 
                  (match bt with
                  | UInt32 | Int32 -> o_str "32"
                  | UInt64 | Int64 -> o_str "64"
                  | _ -> failwith "o_stmt : This was supposed to be an exhaustive match"
                  )
                in o_app f [o_party; o_sz; o_sign; o_ell]
              | Float | Bool -> o_app f [o_party; o_sz])
            in
            seql [o_typ t; o_space; o_expr g e |> init_args] |> s_smln, alb
          else
            seql [o_typ t; o_space; o_expr g e] |> s_smln, alb
    
  | Assign (e1, e2) -> o_codegen_stmt g (Assign_codegen (Base_e e1, Base_e e2))

  | Call (f, args) -> o_with_semicolon (o_app (o_str f) (List.map (o_expr g) args)), g

  | For (_, e1, e2, e3, s) -> o_codegen_stmt g (For_codegen (Base_e e1, Base_e e2, Base_e e3, Base_s s))

  | While (e, s) ->
    o_while (o_expr g e) (o_stmt ([] |> push_local_scope g) s |> fst), g

  | If_else (e, s_then, s_else_opt) ->
     o_codegen_stmt g (If_codegen (Public, Base_e e, Base_s s_then, map_opt s_else_opt (fun s -> Base_s s)))

  | Return eopt -> 
      seql [
        o_str "return ";
        if is_none eopt then o_null else eopt |> get_opt |> o_expr g
      ] |> s_smln, g
    
  | Seq (s1, s2) -> o_codegen_stmt g (Seq_codegen (Base_s s1, Base_s s2))

  | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
     let r, x = get_role e_role, get_var e_var in
     let r1 = role_to_fpstring r in 
     let is_arr = is_array_typ t in
     
     (* bt is the base type and l label *)
     (* let bt, l = get_bt_and_label t |> (fun (bt, l) -> get_inp_type bt, l) in *)
     let bt, l = get_bt_and_label t in
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

     let inp_type = bt |> (if is_secret_label l then basetyp_to_sectyp else basetyp_to_cpptyp)
     in
     let decl_tmp = Line (inp_type ^ " *" ^ tmp_var_name.name ^ " = new " ^ inp_type ^ "[1]")
     in
     
     (* expression that we will initialize each optional loop iteration *)
     let assgn_left =
       Base_e (snd (List.fold_left (fun (i, e) _ ->
                        let i_var = { name = "i" ^ (string_of_int i); index = 0; } in
                        i + 1, Array_read (e, Var i_var |> mk_dsyntax "") |> mk_dsyntax ""
                      ) (0, Var x |> mk_dsyntax "") el))
     in

     (* conditional expression for role == r *)
     let r_cmp =
       let party_var = Var { name = "__party"; index = 0 } |> mk_dsyntax "" in
       let party_const = Var { name = r1 ; index = 0 } |> mk_dsyntax "" in
       Base_e (Binop (Is_equal, party_var, party_const, Some Public) |> mk_dsyntax "")
     in
     
     (* this is the innermost loop body *)
     let base_init =
       (* if role == r then cin into the temporary variable *)
       let tmp_var_name_codegen = Base_e (Var tmp_var_name |> mk_dsyntax "") in
       let tmp_var_name0 = Base_e (Var {name="__tmp_in_" ^ x.name ^ "[0]"; index=0} |> mk_dsyntax "") in 
       let cin = 
        let cin_ = Cin ("cin", tmp_var_name0, bt) in
        if is_secret_label l then If_codegen (Public, r_cmp, cin_, None)
       else cin_ 
       in
       let assgn_right =
        if is_secret_label l then
          let backend_op = basetype_to_secfloat_backend bt in
          let args = [Codegen_String r1; Codegen_String "1"; tmp_var_name_codegen] in
          let args =
            let int_args = 
              match bt with
              | UInt32 -> [Codegen_String "false"; Codegen_String "32"]
              | UInt64 -> [Codegen_String "false"; Codegen_String "64"]
              | Int32 -> [Codegen_String "true"; Codegen_String "32"]
              | Int64 -> [Codegen_String "true"; Codegen_String "64"]
              | _ -> [] in
            args @ int_args in
            App_codegen_expr (backend_op ^ "->input", args)
        else
          tmp_var_name0
       in
       let assgn = Assign_codegen (assgn_left, assgn_right) in
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
    
     let cleanup = Line ("delete[] " ^ tmp_var_name.name) in 
     (* stitch *)
     o_codegen_stmt g (Seq_codegen(Seq_codegen (decl, Seq_codegen (Seq_codegen (print_input_message, decl_tmp), loops)), cleanup))
 
  (* Yes, the last parameter in this type constructor will always be of the form 'Some t' thanks to
    insert_coercions_stmt where Output(r, e, None) is forced to be Output(r, e, typeof_expr e) *)

  | Output (e_role, e, Some t) when is_role e_role ->
     let r = get_role e_role in
     let r1 = role_to_fpstring r in
     let bt, l = get_bt_and_label t in
     let l = l |> get_opt in
     let line = Line ("cout << \"Value of " ^ (expr_to_string e) ^ " : \"") in
     let r_cmp =
        let party_var = Var { name = "__party"; index = 0 } |> mk_dsyntax "" in
        let party_const = Var { name = r1 ; index = 0 } |> mk_dsyntax "" in
     Base_e (Binop (Is_equal, party_var, party_const, Some Public) |> mk_dsyntax "")
     in

     let print_output_msg = if r <> Both then If_codegen (Public, r_cmp, line, None) else line in
     let is_arr = is_array_typ t in
     
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
       aux (
        let get_cout = (fun e ->
          let cout = Cout ("cout", e, bt) in
          if r <> Both then If_codegen (Public, r_cmp, cout, None) else cout
        ) in
        if is_secret_label l then
          let backend, pub = basetype_to_secfloat_backend bt, basetype_to_secfloat_pub bt in
          let publicize = 
            Assign_codegen (Codegen_String pub, App_codegen_expr (backend ^ "->output", [Codegen_String "PUBLIC"; elmt_of_e])) 
          in let display = 
            let out =
              match bt with
              | UInt32 -> ".get_native_type<uint32_t>()[0]"
              | UInt64 -> ".get_native_type<uint64_t>()[0]"
              | Int32 -> ".get_native_type<int32_t>()[0]"
              | Int64 -> ".get_native_type<int64_t>()[0]"
              | Float -> ".get_native_type<float>()[0]"
              | Bool -> ".data[0]"
            in
            Codegen_String ((if bt = Bool then "(bool)" else "" ) ^ pub ^ out) |> get_cout
          in Seq_codegen (publicize, display)
        else
          elmt_of_e |> get_cout
       )
     in
     o_codegen_stmt g (Seq_codegen (print_output_msg, output_gate_loops))

  | Skip s -> (if s = "" then o_null else seq o_newline (seq (o_comment s) o_newline)), g
           
  | _ -> failwith "codegen_stmt: impossible branch"

and o_codegen_stmt (g:gamma) (s:codegen_stmt) :comp * gamma =
  match s with
  | Base_s s -> o_stmt g s

  | App_codegen (f, args) -> o_with_semicolon (o_app (o_str f) (List.map (o_codegen_expr g) args)), g

  | Line s  -> o_str s |> s_smln, g

  | Cin (s, x, _) ->
     (if Config.get_dummy_inputs () then
        o_with_semicolon (seq (o_codegen_expr g x) (o_str (" = rand()")))
      else
        o_with_semicolon (seq (o_str s) (seq (o_str " >> ") (o_codegen_expr g x)))),
     g

  (* Base type is only used in OBLIVC backend *)
  | Cout (s, e, _) ->
     (* o_with_semicolon (seq (o_str s) (seq (o_str " << ") (seq (o_paren (o_codegen_expr g e)) (o_str " << endl")))) *)
     (* Too many brackets, cleaner line below *)
     seql [o_str s; o_str " << "; o_paren @@ o_codegen_expr g e; o_str " << endl"] |> s_smln,
     g

  | Open_file (write, x, f) ->
     (if write then
        let c = o_str ("ofstream " ^ x ^ ";\n" ^ x ^ ".open(\"" ^ f ^ "\", ios::out | ios::trunc);\n") in
        out_files := !out_files @ [x];
        c
      else o_str ("ifstream " ^ x ^ ";\n" ^ x ^ ".open(\"" ^ f ^ "\", ios::in);\n")),
     g

  | Close_file f -> o_str (f ^ ".close ();\n"), g

  | Assign_codegen (e1, e2) -> o_assign (o_codegen_expr g e1) (o_codegen_expr g e2) |> s_smln, g
           
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
     o_ite (o_codegen_expr g e) (s_then |> o_codegen_stmt g_body |> fst)
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

let prelude_string :string = "
#include <vector>\n\
#include <math.h>\n\
#include <cstdlib>\n\
#include <iostream>\n\
#include <fstream>\n\
\n\
using namespace std ;\n\
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
  return std::vector<T>(size) ;\n\
}\n\
\n\
template <typename T, typename... Args>\n\
auto make_vector(size_t first, Args... sizes)\n\
{\n\
  auto inner = make_vector<T>(sizes...) ;\n\
  return vector<decltype(inner)>(first, inner) ;\n\
}\n\
\n
"
                                   
let o_one_program ((globals, main):global list * codegen_stmt) (ofname:string) :unit =
  let prelude = o_str prelude_string
  in
  let main_header = o_str
"\n\nint main (int __argc, char **__argv) {\n\
\n" in
  let main_prelude = o_null in
  
  let main_prelude, g =
    let c_globals, g = List.fold_left (fun (c, g) d ->
                           let c_global, g = o_global g d in
                           seq c (seq o_newline c_global), g) (o_null, empty_env) globals
    in
    seq prelude (seq c_globals (seq main_header main_prelude)), g
  in
  let main_body, g = o_codegen_stmt ([] |> push_local_scope g) main in
  let main_end = o_null in
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