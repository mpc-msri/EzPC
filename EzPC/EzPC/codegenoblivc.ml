(*

Authors: Aseem Rastogi, Lohith Ravuru, Nishant Kumar, Mayank Rathee.

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

open Codegenlib
   
let o_hd_and_args (head:comp) (args:comp list) :comp =
  match args with
  | []     -> seq head (seq (o_str "(") (o_str ")"))
  | hd::tl ->
     let c = seq head (seq (o_str "(") hd) in
     let c = List.fold_left (fun c arg -> seq c (seq (o_str ", ") arg)) c tl in
     seq c (o_str ")")

let o_app (head:comp) (args:comp list) :comp = o_hd_and_args head args

let o_role (r:role) :comp =
  match r with
  | Server -> o_str "2"
  | Client -> o_str "1"
  | Both   -> o_str "0"

let o_var (x:var) :comp = o_str x.name (* o_str (x.name ^ (string_of_int x.index)) *)

let o_unop :unop -> comp = function
  | U_minus -> o_str "-"
  | Bitwise_neg -> o_str "~"
  | Not -> o_str "!"
                        
let o_binop :binop -> comp = function
  | Sum          -> o_str "+"
  | Sub          -> o_str "-"
  | Mul          -> o_str "*"
  | Div          -> o_str "/"
  | Mod          -> o_str "%"
  | Pow          -> failwith "Pow is not an infix operator, so o_pbinop can't handle it"
  | Greater_than -> o_str ">"
  | Less_than -> o_str "<"
  | Is_equal -> o_str "=="
  | Greater_than_equal -> o_str ">="
  | Less_than_equal -> o_str "<="
  | R_shift_a    -> o_str ">>"
  | L_shift      -> o_str "<<"
  | Bitwise_and  -> o_str "&"
  | Bitwise_or   -> o_str "|"
  | Bitwise_xor  -> o_str "^"
  | And          -> o_str "&"
  | Or           -> o_str "|"
  | Xor          -> o_str "^"
  | R_shift_l    -> o_str ">>"

let o_conditional (c1:comp) (c2:comp) (c3:comp) :comp =
  seq c1 (seq (o_str " ? ") (seq c2 (seq (o_str " : ") c3)))

let o_basetyp (t:base_type) :comp =
  match t with
  | UInt32 -> o_str "uint32_t"
  | UInt64 -> o_str "uint64_t"
  | Bool -> o_str "bool"
  | _ -> failwith "Signed int types are not supported in OblivC backend yet"

let o_printf (t:base_type) :comp = 
  match  t with
  | UInt32 -> seq (o_string_literal "%") (o_str " PRIu32 ")
  | UInt64 -> seq (o_string_literal "%") (o_str " PRIu64 ")
  | Bool   -> o_string_literal "%d"
  | _ -> failwith "Signed int types are not supported in OblivC backend yet"

let o_scanf (t:base_type) :comp = 
  match  t with
  | UInt32 -> seq (o_string_literal "%") (o_str " SCNu32 ")
  | UInt64 -> seq (o_string_literal "%") (o_str " SCNu64 ")
  | Bool   -> o_string_literal "%d"
  | _ -> failwith "Signed int types are not supported in OblivC backend yet"

let o_iofunction (i:bool) (t:base_type) :comp =
  seq (if i then o_str "feedObliv" else o_str "revealObliv")
      (match  t with
       | UInt32 -> o_str "Long"
       | UInt64 -> o_str "LLong"
       | Bool -> o_str "Bool"
       | _ -> failwith "Signed int types are not supported in OblivC backend yet")

let o_ite (is_obliv:bool) (c_if:comp) (c_then:comp) (c_else_opt:comp option) :comp =
  let if_str = if is_obliv then "obliv if" else "if" in
  let c_if_then = seq (o_str (if_str ^ " (")) (seq c_if (seq (o_str ") {") (seq o_newline (seq c_then (seq o_newline (o_str "}")))))) in
  if is_none c_else_opt then c_if_then
  else seq c_if_then (seq (o_str " else {") (seq o_newline (seq (get_opt c_else_opt) (seq o_newline (o_str "}")))))

let o_assign (lhs:comp) (rhs:comp) :comp = seq lhs (seq (o_str " = ") rhs)
          
let rec o_expr (e:expr) :comp =
  match e.data with
  | Role r -> o_role r

  | Const (UInt32C n) -> o_uint32 n

  | Const (UInt64C n) -> o_uint64 n

  | Const (BoolC b) -> o_bool b

  | Const _ -> failwith "Signed int types are not supported in OblivC backend yet"
    
  | Var s -> o_var s

  | Unop (op, e, _) -> seq (o_unop op) (seq o_space (o_expr e))

  | Binop (op, e1, e2, _) -> 
      o_paren (match op with
               | R_shift_a -> o_app (o_str "uarshift") [o_expr e1; o_expr e2]
               | Pow -> o_app (o_str "pow") [o_expr e1; o_expr e2]
               | _ -> seq (o_expr e1) (seq o_space (seq (o_binop op) (seq o_space (o_expr e2)))))
  
  | Conditional (e1, e2, e3, Some Public) -> o_codegen_expr (Conditional_codegen (Base_e e1, Base_e e2, Base_e e3, Public))

  | Array_read (e1, e2) -> seq (o_str " *") (seq (o_str "(") (seq (seq ( seq (o_expr e1) (o_str " + ") ) (o_expr e2)) (o_str ")")))

  | App (f, args) -> o_app (o_str f) (List.map o_expr args)

  | Subsumption (e, l1, Secret l2) -> o_expr e

  | _ -> failwith "o_expr: impossible branch"

and o_codegen_expr (e:codegen_expr) :comp =
  match e with
  | Base_e e -> o_expr e
              
  | Input_g (r, sl, s, bt) -> o_app (o_iofunction true bt) [o_var s; o_role r]

  | Dummy_g (sl, _) -> o_str "rand()"
    
  | Conditional_codegen (e1, e2, e3, l) ->
     let c1, c2, c3 = o_codegen_expr e1, o_codegen_expr e2, o_codegen_expr e3 in
     o_conditional c1 c2 c3

  | App_codegen_expr (f, el) -> o_app (o_str f) (List.map o_codegen_expr el)

  | _ -> failwith "o_codegen_expr: impossible case"
                                      
let rec o_typ (t:typ) :comp =
  match t.data with
  | Base (bt, Some (Secret _)) -> seq (o_str "obliv ") (o_basetyp bt)
  | Base (bt, _) -> o_basetyp bt
  | Array (quals, typ, _) ->
     let bt, l = (get_bt_and_label typ) in
     seq (if quals |> List.mem Immutable then o_str "const " else o_null) (seq (o_typ (Base (bt, l) |> mk_dsyntax "")) (o_str "*"))

let o_ret_typ (t:ret_typ) :comp =
  match t with
  | Typ t -> o_typ t
  | Void _ -> o_str "void"

let o_array_init (t:typ) :comp =
  let basetyp_to_string (t:base_type) (l:label) :string =
    let prefix = if l = Public then "" else "obliv " in
    let t_str =
      match t with
      | UInt32 -> "uint32_t"
      | UInt64 -> "uint64_t"
      | Bool   -> "bool"
      | _ -> failwith "Signed int types are not supported in OblivC backend yet"
    in
    prefix ^ t_str
  in
             
  let bt, el = get_array_bt_and_dimensions t in 
  let bt, l = get_bt_and_label t in
  let totalsize = List.fold_left (fun i e ->
                      Binop (Mul, i, e, Some Public) |> mk_dsyntax ""
                    ) (App ("sizeof", [Var {name = basetyp_to_string bt (l |> get_opt); index = 0} |> mk_dsyntax ""]) |> mk_dsyntax "") el
  in
  o_app (o_str "malloc") [o_expr totalsize]
  
let o_for (index:comp) (lower:comp) (upper:comp) (body:comp) :comp =
  let init = seq (o_str "for (uint32_t ") (seq index (seq (o_str " = ") lower)) in
  let term = seq index (seq (o_str " < ") upper) in
  let incr = seq index (o_str "++)") in
  let header = seq init (seq (o_str "; ") (seq term (seq (o_str "; ") (seq incr (o_str "{"))))) in
  seq header (seq o_newline (seq body (seq o_newline (o_str "}"))))

let o_while (guard:comp) (body:comp) :comp =
  seq (o_str "while (") (seq guard (seq (o_str ") {\n") (seq body (o_str "\n}\n"))))
  

let o_decl (c_t:comp) (c_varname:comp) (init_opt:comp option) :comp =
  let c = seq c_t (seq o_space c_varname) in
  if is_none init_opt then c
  else seq c (seq (o_str " = ") (get_opt init_opt))
                                         
let get_fresh_var :(unit -> string) =
  let r = ref 0 in
  fun _ -> let s = "__tmp__fresh_var_" ^ (string_of_int !r) in r := !r + 1; s

let rec o_stmt (s:stmt) :comp =
  match s.data with
  | Decl (t, e, init_opt) ->
     (match init_opt with
      | None ->
         let o_init = if is_array_typ t then Some (o_array_init t) else None in
         o_with_semicolon (o_decl (o_typ t) (o_expr e) o_init)
      | Some init ->
         o_stmt (Seq (Decl   (t, e, None) |> mk_dsyntax "",
                      Assign (e, init)    |> mk_dsyntax "") |> mk_dsyntax ""))

  | Assign (e, { data = Conditional (e1, e2, e3, Some l) }) when is_secret_label l ->
     o_codegen_stmt (If_codegen (l, Base_e e1, Assign_codegen (Base_e e, Base_e e2),
                                 Some (Assign_codegen (Base_e e, Base_e e3))))

  | Assign (e, e1) -> o_with_semicolon (o_codegen_stmt (Assign_codegen (Base_e e, Base_e e1)))

  | Call (f, args) -> o_with_semicolon (o_app (o_str f) (List.map o_expr args))

  | For (_, e1, e2, e3, s) -> o_codegen_stmt (For_codegen (Base_e e1, Base_e e2, Base_e e3, Base_s s))

  | While (e, s) -> o_while (o_expr e) (o_stmt s)

  | If_else (e, s_then, s_else_opt) -> o_codegen_stmt (If_codegen (Public, Base_e e, Base_s s_then, map_opt s_else_opt (fun s -> Base_s s)))

  | Return eopt -> o_with_semicolon (seq (o_str "return ")
                                         (if is_none eopt then o_null else eopt |> get_opt |> o_expr))
    
  | Seq (s1, s2) -> o_codegen_stmt (Seq_codegen (Base_s s1, Base_s s2))

  | Input (e_role, e_var, t) when is_role e_role && is_var e_var ->
     let rng = s.metadata in
     let r, x = get_role e_role, get_var e_var in
     let is_arr = is_array_typ t in
     
     (* bt is the base type and l label *)
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
       let role_var = { name = "role"; index = 0 } in
       let r_cmp = Base_e (Binop (Is_equal, Var role_var |> mk_dsyntax "", Role r |> mk_dsyntax "", Some Public) |> mk_dsyntax "") in
       (* if role == r then cin into the temporary variable *)
       let cin = Cin ("scanf", Base_e (Var tmp_var_name |> mk_dsyntax "") , bt) in

       if is_secret_label l then
         let sl = get_secret_label l in
         let cin = If_codegen (Public, r_cmp, cin, None) in
         (* add an input gate *)
         let assgn = Assign_codegen (assgn_left, Input_g (r, sl, tmp_var_name, bt))
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
     o_codegen_stmt (Seq_codegen (decl, Seq_codegen (decl_tmp, loops)))

  | Output (e_role, e, Some t) when is_role e_role ->
     let r = get_role e_role in
     let bt, l = get_bt_and_label t in
     if not (l |> get_opt |> is_secret_label) then o_codegen_stmt (Cout ("printf", Base_e e, bt))
     else
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
         aux (Output_s (r, App_codegen_expr ("add_to_output_queue", [Base_e (Var {name = "io"; index = 0} |> mk_dsyntax "");
                                                                     Base_e (Role r |> mk_dsyntax "")]),
                        elmt_of_e, bt))
       in

       o_codegen_stmt output_gate_loops
           
  | _ -> failwith "codegen_stmt: impossible branch"

and o_codegen_stmt (s:codegen_stmt) :comp =
  match s with
  | Base_s s -> o_stmt s

  | App_codegen (f, args) -> o_with_semicolon (o_app (o_str f) (List.map o_codegen_expr args))

  | Cin (s, x, bt) ->
     if Config.get_dummy_inputs () then
       o_with_semicolon (seq (o_codegen_expr x) (o_str (" = rand()")))
     else
       o_with_semicolon ( o_app (o_str s) [ (o_scanf bt) ; seq (o_str "&") (o_codegen_expr x) ] )

  | Cout (s, e, bt) -> o_with_semicolon ( o_app (o_str s) [ (o_printf bt) ; o_codegen_expr e] )

  | Assign_codegen (e1, e2) -> o_with_semicolon (o_assign (o_codegen_expr e1) (o_codegen_expr e2))

  | For_codegen (e1, e2, e3, s) -> o_for (o_codegen_expr e1) (o_codegen_expr e2) (o_codegen_expr e3) (o_codegen_stmt s)

  | If_codegen (l, e, s_then, s_else_opt) ->
     o_ite (is_secret_label l) (o_codegen_expr e) (o_codegen_stmt s_then) (map_opt s_else_opt o_codegen_stmt)

  | Seq_codegen (s1, s2) -> seq (o_codegen_stmt s1) (seq o_newline (o_codegen_stmt s2))

  | Output_s (r, e1, e2, bt) -> o_with_semicolon (o_app (o_iofunction false bt) [o_codegen_expr e1; o_codegen_expr e2; o_role r])

  | _ -> failwith "o_codegen_stmt: impossible case"

let o_binder (b:binder) :comp = seq (b |> snd |> o_typ) (seq o_space (b |> fst |> o_var))

let o_global (d:global) :comp =
  match d.data with
  | Fun (_, fname, bs, body, ret_t) ->
     let header = seq (o_ret_typ ret_t) (seq o_space (o_hd_and_args (o_str fname) (List.map o_binder bs))) in
     seq header (seq (o_str " {\n") (seq (o_stmt body) (o_str "\n}\n")))
  | Extern_fun (_, fname, bs, ret_t) ->
     let header = seq (o_ret_typ ret_t) (seq o_space (o_hd_and_args (o_str fname) (List.map o_binder bs))) in
     o_with_semicolon (seq (o_str "extern ") header)
  | Global_const (t, e_var, init) -> seq (o_with_semicolon (seq (o_str "const ") (o_decl (o_typ t) (o_expr e_var) (Some (o_expr init))))) o_newline

let c_prelude_string : string = 
"\
/*\n\
This is an autogenerated file, generated using the EzPC compiler.\n\
*/\n\
#include <stdio.h>\n\
#include <stdlib.h>\n\
"

let oblivc_prelude_string : string =
"\
#include <obliv.oh>\n\
#include \"ezpc.h\"\n\
int role;\n\
"

let oblivc_main_prelude_string : string = 
"\
protocolIO *io = arg;\n\
role = io->role;\n\
"

let oblivc_main_epilog_string : string =
"\
\n\
io->gatecount = yaoGateCount();\n\
"

let o_one_program ((globals, main):global list * codegen_stmt) (ofname:string) :unit =
  let prelude =
    o_str (c_prelude_string ^ "\n" ^ oblivc_prelude_string)
  in

  let main_header =
    o_str "\n\nvoid ezpc_main(void* arg) {\n"
  in
  
  let main_prelude = 
    seq (o_str oblivc_main_prelude_string) o_newline
  in

  let main_prelude =
    seq prelude (seq (List.fold_left (fun c d -> seq c (seq o_newline (o_global d))) o_null globals) (seq main_header main_prelude))
  in

  let main_body = o_codegen_stmt main in
  let main_end =
    seq (o_str oblivc_main_epilog_string) (o_str "\n}\n")
  in

  let file = seq main_prelude (seq main_body main_end) in
  
  let b = Buffer.create 0 in
  file b;
  let fd = open_out ofname in
  Buffer.contents b |> fprintf fd "%s\n";
  close_out fd

let o_program ((globals, mains):codegen_program) (ofname_prefix:string) :unit =
  mains |> List.fold_left (fun i m ->
               o_one_program (globals, m) (ofname_prefix ^ (string_of_int i) ^ ".oc");
               i + 1) 0 |> ignore
