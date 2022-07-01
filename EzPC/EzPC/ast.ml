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
open Stdint
open Utils
open Global

type 'a syntax = {
  data : 'a;
  metadata: Metadata.metadata;
}

let mk_syntax (md:Metadata.metadata) (d:'a) :'a syntax = { data = d; metadata = md }

let mk_dsyntax (s:string) (d:'a) :'a syntax = { data = d; metadata = dmeta s }
                                            
type role =
  | Server
  | Client
  | Both
  
type base_type = 
  | UInt32
  | UInt64
  | Int32
  | Int64
  | Bool

type secret_label = 
  | Arithmetic
  | Boolean

type label =
  | Public
  | Secret of secret_label

(* the uint32 and uint64 types come from the stdint package *)
type const =
  | UInt32C  of uint32
  | UInt64C  of uint64
  | Int32C   of int32
  | Int64C   of int64
  | BoolC    of bool

type unop =
  (* Arithmetic *)
  | U_minus | Bitwise_neg
  (* Logical *)
  | Not

type binop =
  (* Arithmetic *)
  | Sum | Sub | Mul | Div | Mod | Pow | Greater_than | Less_than | Is_equal | Greater_than_equal | Less_than_equal | R_shift_a | L_shift | Bitwise_and | Bitwise_or | Bitwise_xor
  (* Logical *)
  | And | Or | Xor | R_shift_l

type var = {
    name: string;
    index: int
  }

type qualifier =
  | Partition    (* functions having this qualifier get their top-level applications partitioned *)
  | Inline       (* functions having this qualifier get their applications inlined *)
  | Unroll       (* for loops that should be unrolled statically *)
  | Immutable    (* array types can be immutable *)
  | Extern       (* extern function declaration *)

type qualifiers = qualifier list
  
type expr' =
  | Role          of role
  | Const	  of const
  | Var 	  of var
  | Unop          of unop * expr * label option
  | Binop 	  of binop * expr * expr * label option
  | Conditional   of expr  * expr * expr * label option
  | Array_read	  of expr * expr
  | App           of string * expr list
  | Subsumption	  of expr * label * label

and expr = expr' syntax
                   
and typ' =
  | Base  of base_type * label option
  | Array of qualifiers * typ * expr                       (* array element type, length *)

and typ = typ' syntax

type ret_typ =
  | Typ of typ
  | Void of unit syntax

type stmt' = 
  | Decl    of typ * expr * expr option                    (* type, variable name, optional initializer *)
  | Assign  of expr * expr                                 (* lhs, rhs *)
  | Call    of string * expr list                          (* function call *)
  | For     of qualifiers * expr * expr * expr * stmt      (* index, lower, upper (non inclusive), body *)
  | While   of expr * stmt                                 (* guard, body *)
  | If_else of expr * stmt * stmt option                   (* conditional, then, optional else *)
  | Return  of expr option                                 (* return expression *)
  | Seq	    of stmt * stmt
  | Input   of expr * expr * typ                           (* role, variable name, type *)
  | Output  of expr * expr * typ option                    (* role, expression to output, type *)
  | Skip    of string                                      (* dummy statement, repurpose it for different metadata *)

and stmt = stmt' syntax

type binder = var * typ                                                  (* binder name, binder type *)

type global' =
  | Fun of qualifiers * string * binder list * stmt * ret_typ (* function name, parameters, body, return type *)
  | Extern_fun of qualifiers * string * binder list * ret_typ (* function name, parameters, return type - no body *)
  | Global_const of typ * expr * expr                         (* type, var, initializer *)

type global = global' syntax
type program = global list

let label_to_string (l:label) :string =
  match l with
  | Public -> "public"
  | Secret Arithmetic -> "arithmetic"
  | Secret Boolean -> "boolean"

let role_to_string (r:role) :string =
  match r with
  | Server -> "SERVER"
  | Client -> "CLIENT"
  | Both   -> "ALL"

let unop_to_string (u:unop) :string =
  match u with
  | U_minus -> "-"
  | Bitwise_neg -> "~"
  | Not -> "!"
            
let binop_to_string (b:binop) :string =
  match b with
  | Sum -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | Mod -> "%"
  | Pow -> "pow"
  | Greater_than -> ">"
  | Less_than -> "<"
  | Is_equal -> "=="
  | Greater_than_equal -> ">="
  | Less_than_equal -> "<="
  | R_shift_a -> ">>"
  | L_shift -> "<<"
  | Bitwise_and -> "&"
  | Bitwise_or -> "|"
  | Bitwise_xor -> "^"
  | And -> "&&"
  | Or -> "||"
  | Xor -> "xor"
  | R_shift_l -> ">>>"
            
let rec expr_to_string (e:expr) :string =
  match e.data with
  | Role r -> role_to_string r
  | Const (UInt32C n) -> Uint32.to_string n
  | Const (UInt64C n) -> Uint64.to_string n
  | Const (Int32C n)  -> Int32.to_string n
  | Const (Int64C n)  -> Int64.to_string n
  | Const (BoolC b)   -> string_of_bool b
  | Var x -> x.name
  | Unop (op, e, lopt) ->
     let op_str = unop_to_string op ^ "_" ^
                    if is_none lopt then "<no label>" else lopt |> get_opt |> label_to_string
     in
     op_str ^ " " ^ expr_to_string e
  | Binop (op, e1, e2, lopt) ->
     let op_str =
       binop_to_string op ^ "_" ^
         if is_none lopt then "<no label>" else label_to_string (get_opt lopt)
     in
     expr_to_string e1 ^ " " ^ op_str ^ " " ^ expr_to_string e2
  | Conditional (e1, e2, e3, lopt) ->
     expr_to_string e1 ^ " ?_" ^ (if is_none lopt then "<no label>" else label_to_string (get_opt lopt)) ^
       " " ^ expr_to_string e2 ^ " : " ^ expr_to_string e3
  | Array_read (e1, e2) -> expr_to_string e1 ^ "[" ^ expr_to_string e2 ^ "]"
  | App (f, args) -> f ^ "(" ^
                       if List.length args = 0 then ")"
                       else
                         (List.fold_left (fun s arg -> s ^ ", " ^ expr_to_string arg) (expr_to_string (List.hd args)) (List.tl args)) ^ ")"
  | Subsumption (e, src, tgt) -> "<" ^ label_to_string src ^ " ~> " ^ label_to_string tgt ^ "> " ^ expr_to_string e

let rec typ_to_string (t:typ) :string =
  match t.data with
  | Base (bt, lopt) ->
     let prefix = if is_none lopt then "<no label>" else label_to_string (get_opt lopt) in
     let bt_str = match bt with
       | UInt32 -> "uint32"
       | UInt64 -> "uint64"
       | Int32  -> "int32"
       | Int64  -> "int64"
       | Bool -> "bool"
     in
     prefix ^ " " ^ bt_str
  | Array (quals, t, e) ->
     let qual_string = if quals |> List.mem Immutable then "const " else "" in
     qual_string ^ typ_to_string t ^ "[" ^ expr_to_string e ^ "]"

let ret_typ_to_string (t:ret_typ) :string =
  match t with
  | Typ t -> t |> typ_to_string
  | Void _ -> "void"
            
let rec stmt_to_string (s:stmt) :string =
  match s.data with
  | Decl (t, e, init_opt) ->
     let decl = typ_to_string t ^ " " ^ expr_to_string e in
     let decl = if is_none init_opt then decl else decl ^ " = " ^ (expr_to_string (get_opt init_opt)) in
     decl ^ ";"
  | Assign (e1, e2) -> expr_to_string e1 ^ " = " ^ expr_to_string e2 ^ ";"
  | Call (f, args) -> "f " ^ "(" ^ (List.fold_left (fun s arg -> s ^ ", " ^ expr_to_string arg) "" args) ^ ")"
  | For (_, e1, e2, e3, s) ->
     "For (" ^ expr_to_string e1 ^ ", " ^ expr_to_string e2 ^ ", " ^ expr_to_string e3 ^ ", " ^ stmt_to_string s ^ ")"
  | While (e, s) -> "While (" ^ expr_to_string e ^ ", " ^ stmt_to_string s ^ ")"
  | If_else (e, then_s, else_s_opt) ->
     "If(" ^ expr_to_string e ^ ", " ^ stmt_to_string then_s ^ ", " ^
       (if is_none else_s_opt then "<no else>" else stmt_to_string (get_opt else_s_opt)) ^ ")"
  | Return eopt -> "return " ^ (if eopt = None then "" else eopt |> get_opt |> expr_to_string)
  | Seq (s1, s2) -> stmt_to_string s1 ^ "\n" ^ stmt_to_string s2
  | Input (e1, e2, t) -> "Input (" ^ expr_to_string e1 ^ ", " ^ expr_to_string e2 ^ ", " ^ typ_to_string t ^ ");"
  | Output (e1, e2, topt) ->
     "Output (" ^ expr_to_string e1 ^ ", " ^ expr_to_string e2 ^ ", " ^
       (if is_none topt then "<no typ>" else typ_to_string (get_opt topt)) ^ ");"
  | Skip s -> "Skip: " ^ s

let is_array_typ (t:typ) :bool =
  match t.data with
  | Array _ -> true
  | _ -> false

let get_typ_quals (t:typ) :qualifiers =
  match t.data with
  | Base _ -> []
  | Array (quals, _, _) -> quals
       
let rec get_bt_and_label (t:typ) :(base_type * label option) =
  match t.data with
  | Base (bt, l) -> bt, l
  | Array (_, t, _) -> get_bt_and_label t

let rec get_base_lvalue (e:expr) :var =
  match e.data with
  | Var x -> x
  | Array_read (e1, e2) -> get_base_lvalue e1
  | _ -> failwith ("get_base_lvalue: expression " ^ (expr_to_string e) ^ " is not an lvalue")

let rec change_array_read_var (e:expr) (new_var:var) :expr =
  match e.data with
  | Var x -> Var new_var |> mk_dsyntax ""
  | Array_read (e1, e2) -> Array_read ((change_array_read_var e1 new_var), e2) |> mk_dsyntax ""
  | _ -> failwith "wrong type encountered"

let is_secret_label (l:label) :bool =
  match l with
  | Secret _ -> true
  | _ -> false

let get_secret_label (l:label) :secret_label =
  match l with
  | Secret s -> s
  | _ -> failwith "get_secret_label: label is not secret"

let get_array_bt_and_dimensions (t:typ) :(typ * expr list) =
  let rec aux (t:typ) (l:expr list) :(typ * expr list) =
    match t.data with
    | Base _ -> t, List.rev_append l []
    | Array (_, t, e) -> aux t (e::l)
  in
  aux t []

let rec get_array_flat_size (t:typ) :expr = 
  match t.data with
    | Base _ -> (Const (Int32C 1l) |> mk_dsyntax "")
    | Array (_, t, e) -> (Binop (Mul, (get_array_flat_size t), e, Some Public)) |> mk_dsyntax ""

let is_const (e:expr) :bool =
  match e.data with
  | Const _ -> true
  | _ -> false
  
let is_var (e:expr) :bool =
  match e.data with
  | Var _ -> true
  | _ -> false

let get_var (e:expr) :var =
  match e.data with
  | Var x -> x
  | _ -> failwith "get_var: not a Var"

let get_var_name (e:expr) :string =
  match e.data with
  | Var x -> x.name
  | _ -> failwith "get_var: not a Var"
       
let is_role (e:expr) :bool =
  match e.data with
  | Role _ -> true
  | _ -> false

let get_role (e:expr) :role =
  match e.data with
  | Role r -> r
  | _ -> failwith "get_role: not a Role"

let is_pow_2 (e:expr) :bool =
  match e.data with
  | Binop (Pow, { data = Const (UInt32C x) }, _, _) -> Uint32.to_int x = 2
  | Binop (Pow, { data = Const (UInt64C x) }, _, _) -> Uint64.to_int x = 2
  | _ -> false

let get_pow_2_exponent (e:expr) :expr =
  let err () = failwith ("get_pow_2_exponent: " ^ expr_to_string e ^ " not a power of 2") in
  match e.data with
  | Binop (Pow, { data = Const (UInt32C x) }, e2, _) -> 
    if Uint32.to_int x = 2 then e2 else err ()
  | Binop (Pow, { data = Const (UInt64C x) }, e2, _) -> 
    if Uint64.to_int x = 2 then e2 else err ()
  | _ -> err ()

let zero_expr (r:range) :expr = Const (UInt32C Uint32.zero) |> mk_syntax r
let one_expr (r:range) :expr = Const (UInt32C Uint32.one) |> mk_syntax r
let true_expr (r:range) :expr = Const (BoolC true) |> mk_syntax r
let false_expr (r:range) :expr = Const (BoolC false) |> mk_syntax r

let typeof_role (r:range) :typ = Base (UInt32, Some Public) |> mk_syntax r
                               
let typeof_const (c:const) (r:range) :typ =
  (match c with
   | UInt32C n -> Base (UInt32, Some Public)
   | UInt64C n -> Base (UInt64, Some Public)
   | Int32C n  -> Base (Int32, Some Public)
   | Int64C n  -> Base (Int64, Some Public)
   | BoolC b   -> Base (Bool, Some Public)) |> mk_syntax r

let join_types (t1:typ) (t2:typ) :typ option =
  match t1.data, t2.data with
  | Base (UInt32, Some Public), Base (UInt64, Some Public)
  | Base (Int32, Some Public), Base (Int64, Some Public) -> Some t2
  | Base (UInt64, Some Public), Base (UInt32, Some Public)
  | Base (Int64, Some Public), Base (Int32, Some Public) -> Some t1
  | Base (bt1, l1), Base (bt2, l2) when bt1 = bt2 && l1 = l2 -> Some t1
  | _, _ -> None

let typ_of_ret_typ (t:ret_typ) :typ option =
  match t with
  | Typ t -> Some t
  | Void _ -> None

let rec erase_labels_expr (e:expr) :expr =
  let l = Some Public in
  let aux (e:expr') :expr' =
    match e with
    | Role _ -> e
    | Const _ -> e
    | Var _ -> e
    | Unop (op, e, _) -> Unop (op, erase_labels_expr e, l)
    | Binop (op, e1, e2, _) -> Binop (op, erase_labels_expr e1, erase_labels_expr e2, l)
    | Conditional (e1, e2, e3, _) -> Conditional (erase_labels_expr e1, erase_labels_expr e2, erase_labels_expr e3, l)
    | Array_read (e1, e2) -> Array_read (erase_labels_expr e1, erase_labels_expr e2)
    | App (f, el) -> App (f, el |> List.map erase_labels_expr)
    | Subsumption (e, _, _) -> (erase_labels_expr e).data
  in
  { e with data = aux e.data }

let rec erase_labels_typ (t:typ) :typ =
  let l = Some Public in
  let aux (t:typ') :typ' =
    match t with
    | Base (bt, _) -> Base (bt, l)
    | Array (quals, t, e) -> Array (quals, erase_labels_typ t, erase_labels_expr e)
  in
  { t with data = aux t.data }

let erase_labels_ret_typ (t:ret_typ) :ret_typ =
  match t with
  | Typ t -> Typ (t |> erase_labels_typ)
  | Void _ -> t
            
let rec erase_labels_stmt (s:stmt) :stmt =
  let aux (s:stmt') :stmt' =
    match s with
    | Decl (t, e, eopt) -> Decl (erase_labels_typ t, erase_labels_expr e, map_opt eopt erase_labels_expr)
    | Assign (e1, e2) -> Assign (erase_labels_expr e1, erase_labels_expr e2)
    | Call (f, el) -> Call (f, el |> List.map erase_labels_expr)
    | For (quals, e1, e2, e3, s) -> For (quals, erase_labels_expr e1, erase_labels_expr e2, erase_labels_expr e3, erase_labels_stmt s)
    | While (e, s) -> While (erase_labels_expr e, erase_labels_stmt s)
    | If_else (e, s_then, s_else_opt) -> If_else (erase_labels_expr e, erase_labels_stmt s_then, map_opt s_else_opt erase_labels_stmt)
    | Return eopt -> Return (map_opt eopt erase_labels_expr)
    | Seq (s1, s2) -> Seq (erase_labels_stmt s1, erase_labels_stmt s2)
    | Input (e1, e2, t) -> Input (erase_labels_expr e1, erase_labels_expr e2, erase_labels_typ t)
    | Output (e1, e2, topt) -> Output (erase_labels_expr e1, erase_labels_expr e2, map_opt topt erase_labels_typ)
    | Skip _ -> s
  in
  { s with data = aux s.data }

let erase_labels_binder (b:binder) :binder = (fst b, snd b |> erase_labels_typ)

let erase_labels_global (d:global) :global =
  let aux (d:global') :global' =
    match d with
    | Fun (quals, fname, bs, body, ret_t) -> Fun (quals, fname, bs |> List.map erase_labels_binder, body |> erase_labels_stmt, ret_t |> erase_labels_ret_typ)
    | Extern_fun (quals, fname, bs, ret_t) -> Extern_fun (quals, fname, bs |> List.map erase_labels_binder, ret_t |> erase_labels_ret_typ)
    | Global_const (t, e_var, init) -> Global_const (t |> erase_labels_typ, e_var |> erase_labels_expr, init |> erase_labels_expr)
  in
  { d with data = aux d.data }

let erase_labels_program (p:program) :program = p |> List.map erase_labels_global
  
module SSet = Set.Make (
                  struct
                    let compare = Pervasives.compare
                    type t = var
                  end
                )

(*
 * arrays are passed by reference
 * so a function call modifies the array arguments, unless qualifier with Immutable
 *)
let rec modifies_expr (g:(string * binder list) list) (e:expr) :SSet.t =
  let aux (el:expr list) :SSet.t = el |> List.fold_left (fun s e -> SSet.union s (e |> modifies_expr g)) SSet.empty
  in
  match e.data with
  | Role _ | Const _ | Var _ -> SSet.empty
  | Unop (_, e, _) -> e |> modifies_expr g
  | Binop (_, e1, e2, _) -> aux [e1; e2]
  | Conditional (e1, e2, e3, _) -> aux [e1; e2; e3]
  | Array_read (_, e2) -> e2 |> modifies_expr g
  | App (f, args) ->
     let s1 = aux args in
     let bs = g |> List.find (fun (f', _) -> f = f') |> snd in
     let arr_args = List.fold_left2 (fun arr_args arg (_, t) ->
                        if is_array_typ t && not (get_typ_quals t |> List.mem Immutable)
                        then arr_args @ [arg]
                        else arr_args) [] args bs
     in
     SSet.union s1 (arr_args |> List.map get_base_lvalue |> SSet.of_list)
  | Subsumption (e, _, _) -> e |> modifies_expr g

let rec modifies_typ (g:(string * binder list) list) (t:typ) :SSet.t =
  match t.data with
  | Base _ -> SSet.empty
  | Array (_, bt, e) -> SSet.union (modifies_typ g bt) (modifies_expr g e)
                   
let modifies (g:(string * binder list) list) (s:stmt) :SSet.t =
  let rec modifies_and_defined (s:stmt) :(SSet.t * SSet.t) =
    match s.data with
    | Decl (t, e_var, init_opt) when is_var e_var ->
       SSet.union (modifies_typ g t)
                  (if init_opt = None then SSet.empty else init_opt |> get_opt |> modifies_expr g),
       SSet.singleton (get_var e_var)

    | Assign (e1, e2) ->
       SSet.union (SSet.singleton (get_base_lvalue e1))
                  (SSet.union (modifies_expr g e1) (modifies_expr g e2)),
       SSet.empty

    | Call (f, args) ->
       let s1 = args |> List.fold_left  (fun s arg -> SSet.union s (modifies_expr g arg)) SSet.empty in
       let bs = g |> List.find (fun (f', _) -> f = f') |> snd in
       let arr_args = List.fold_left2 (fun arr_args arg (_, t) ->
                          if is_array_typ t && not (get_typ_quals t |> List.mem Immutable)
                          then arr_args @ [arg]
                          else arr_args) [] args bs
       in
       SSet.union s1 (arr_args |> List.map get_base_lvalue |> SSet.of_list),
       SSet.empty
       
    | For (_, e_var, e1, e2, s) when is_var e_var ->
       let s1, s2 = modifies_and_defined s in
       SSet.union (SSet.remove (get_var e_var) (SSet.diff s1 s2))
                  (SSet.union (modifies_expr g e1) (modifies_expr g e2)),
       SSet.empty

    | While (e, s) ->
       let m_s, d_s = modifies_and_defined s in
       SSet.union (modifies_expr g e) m_s, SSet.empty

    | If_else (e, s_then, s_else_opt) ->
       let m_then, d_then = modifies_and_defined s_then in
       let m_else, d_else =
         if is_none s_else_opt then SSet.empty, SSet.empty
         else modifies_and_defined (get_opt s_else_opt)
       in
       SSet.union (SSet.union (SSet.diff m_then d_then) (SSet.diff m_else d_else))
                  (modifies_expr g e),
       SSet.empty

    | Return eopt ->
       (if is_none eopt then SSet.empty else eopt |> get_opt |> modifies_expr g),
       SSet.empty

    | Seq (s1, s2) ->
       let m1, d1 = modifies_and_defined s1 in
       let m2, d2 = modifies_and_defined s2 in
       SSet.union m1 m2, SSet.union d1 d2
       
    | Output (e_role, e, topt) when is_role e_role ->
       SSet.union (modifies_expr g e)
                  (if topt = None then SSet.empty else topt |> get_opt |> modifies_typ g),
       SSet.empty

    | Input (e_role, e_var, t) when is_var e_var && is_role e_role ->
       modifies_typ g t, SSet.singleton (get_var e_var)

    | Skip _ -> SSet.empty, SSet.empty

    | _ -> failwith ("modifies: unexpteced statement: " ^ stmt_to_string s)

  in
  let m, d = modifies_and_defined s in
  SSet.diff m d

let rec get_vars_in_expr (e:expr) :SSet.t =
  match e.data with
  | Role _ -> SSet.empty
  | Const _ -> SSet.empty
  | Var x -> SSet.singleton x
  | Unop (_, e, _) -> get_vars_in_expr e
  | Binop (_, e1, e2, _) -> SSet.union (get_vars_in_expr e1) (get_vars_in_expr e2)
  | Conditional (e1, e2, e3, _) -> SSet.union (SSet.union (get_vars_in_expr e1) (get_vars_in_expr e2))
                                              (get_vars_in_expr e3)
  | Array_read (e1, e2) -> SSet.union (get_vars_in_expr e1) (get_vars_in_expr e2)
  | App (_, args) -> args |> List.fold_left (fun s arg -> SSet.union s (get_vars_in_expr arg)) SSet.empty
  | Subsumption (e, _, _) -> get_vars_in_expr e
                           
let rec eq_expr (e1:expr) (e2:expr) :bool =
  match e1.data, e2.data with
  | Role r1, Role r2 -> r1 = r2
  | Const c1, Const c2 -> c1 = c2
  | Var x1, Var x2 -> x1 = x2
  | Unop (op1, e1, lopt1), Unop (op2, e2, lopt2) ->
     op1 = op2 && lopt1 = lopt2 && eq_expr e1 e2
  | Binop (op1, e11, e12, lopt1), Binop (op2, e21, e22, lopt2) ->
     op1 = op2 && lopt1 = lopt2 && eq_expr e11 e21 && eq_expr e12 e22
  | Conditional (e11, e12, e13, lopt1), Conditional (e21, e22, e23, lopt2) ->
     lopt1 = lopt2 && eq_expr e11 e21 && eq_expr e12 e22 && eq_expr e13 e23
  | Array_read (e11, e12), Array_read (e21, e22) -> eq_expr e11 e21 && eq_expr e12 e22
  | App (f1, args1), App (f2, args2) ->
     f1 = f2 && List.length args1 = List.length args2 && List.for_all2 (fun e1 e2 -> eq_expr e1 e2) args1 args2
  | Subsumption (e1, src1, tgt1), Subsumption (e2, src2, tgt2) ->
     src1 = src2 && tgt1 = tgt2 && eq_expr e1 e2
  | _, _ -> false

(*
 * substitute e_from by e_to in e
 *)
let rec subst_expr (e_from:expr) (e_to:expr) (e:expr) :expr =
  let subst = subst_expr e_from e_to in
  let aux (e:expr') :expr'=
    match e with
    | Role _ -> e
    | Const _ -> e
    | Var x -> if is_var e_from && get_var e_from = x then e_to.data else e
    | Unop (op, e, lopt) -> Unop (op, subst e, lopt)
    | Binop (op, e1, e2, lopt) -> Binop (op, subst e1, subst e2, lopt)
    | Conditional (e1, e2, e3, lopt) -> Conditional (subst e1, subst e2, subst e3, lopt)
    | Array_read (e1, e2) -> Array_read (subst e1, subst e2)
    | App (f, args) -> App (f, args |> List.map subst)
    | Subsumption (e, src, tgt) -> Subsumption (subst e, src, tgt)
  in
  if eq_expr e e_from then e_to
  else e.data |> aux |> (fun e' -> { e with data = e' })

let rec subst_typ (e_from:expr) (e_to:expr) (t:typ) :typ =
  let aux (t:typ') :typ' =
    match t with
    | Base _ -> t
    | Array (quals, bt, e) -> Array (quals, subst_typ e_from e_to bt, subst_expr e_from e_to e)
  in
  t.data |> aux |> (fun t' -> { t with data = t' })
  
let rec subst_stmt (e_from:expr) (e_to:expr) (s:stmt) :stmt =
  let subst = subst_stmt e_from e_to in
  let subst_t = subst_typ e_from e_to in
  let subst_e = subst_expr e_from e_to in
  let aux (s:stmt') :stmt' =
    match s with
    | Decl (t, e, eopt) -> Decl (subst_t t, subst_e e, map_opt eopt subst_e)
    | Assign (e1, e2) -> Assign (subst_e e1, subst_e e2)
    | Call (f, args) -> Call (f, args |> List.map subst_e)
    | For (quals, e_var, e1, e2, body) -> For (quals, e_var, subst_e e1, subst_e e2, subst body)
    | While (e, s) -> While (subst_e e, subst s)
    | If_else (e, s_then, s_else_opt) -> If_else (subst_e e, subst s_then, map_opt s_else_opt subst)
    | Return eopt -> Return (map_opt eopt subst_e)
    | Seq (s1, s2) -> Seq (subst s1, subst s2)
    | Input (e_role, e_var, t) -> Input (e_role, e_var, subst_t t)
    | Output (e_role, e, topt) -> Output (e_role, subst_e e, map_opt topt subst_t)
    | Skip _ -> s
  in
  s.data |> aux |> (fun s' -> { s with data = s' })

let rec var_used_in_expr (e:expr) (x:var) :bool =
  let aux (e:expr') :bool =
    match e with
    | Role _ | Const _ -> false
    | Var y -> x = y
    | Unop (_, e, _) -> x |> var_used_in_expr e
    | Binop (_, e1, e2, _) -> x |> var_used_in_expr e1 || x |> var_used_in_expr e2
    | Conditional (e, e1, e2, _) -> x |> var_used_in_expr e || x |> var_used_in_expr e1 || x |> var_used_in_expr e2
    | Array_read (e1, e2) -> x |> var_used_in_expr e1 || x |> var_used_in_expr e2
    | App (_, el) -> el |> List.exists (fun e -> x |> var_used_in_expr e)
    | Subsumption (e, _, _) -> x |> var_used_in_expr e
  in
  aux e.data

let rec var_used_in_typ (t:typ) (x:var) :bool =
  let aux (t:typ') :bool =
    match t with
    | Base _ -> false
    | Array (_, t, e) -> x |> var_used_in_typ t || x |> var_used_in_expr e
  in
  aux t.data

let var_used_in_ret_typ (t:ret_typ) (x:var) :bool =
  match t with
  | Typ t -> x |> var_used_in_typ t
  | Void _ -> false

let rec var_used_in_stmt (s:stmt) (x:var) :bool =
  let aux (s:stmt') :bool =
    match s with
    | Decl (t, e_var, eopt) -> x |> var_used_in_typ t || (is_some eopt && x |> var_used_in_expr (eopt |> get_opt))
    | Assign (e1, e2) -> x |> var_used_in_expr e1 || x |> var_used_in_expr e2
    | Call (_, el) -> el |> List.exists (fun e -> x |> var_used_in_expr e)
    | For (_, _, e_lower, e_upper, s) -> x |> var_used_in_expr e_lower || x |> var_used_in_expr e_upper || x |> var_used_in_stmt s
    | While (e, s) -> x |> var_used_in_expr e || x |> var_used_in_stmt s
    | If_else (e, s_then, s_else_opt) ->
       x |> var_used_in_expr e || x |> var_used_in_stmt s_then || (is_some s_else_opt && x |> var_used_in_stmt (s_else_opt |> get_opt))
    | Return eopt -> is_some eopt && x |> var_used_in_expr (eopt |> get_opt)
    | Seq (s1, s2) -> x |> var_used_in_stmt s1 || x |> var_used_in_stmt s2
    | Input (_, _, t) -> x |> var_used_in_typ t
    | Output (_, e, topt) -> x |> var_used_in_expr e || (is_some topt && x |> var_used_in_typ (topt |> get_opt))
    | Skip _ -> false
  in
  aux s.data  

let get_unsigned (bt:base_type) :base_type =
  match bt with
  | UInt32 | UInt64 | Bool -> bt
  | Int32 -> UInt32
  | Int64 -> UInt64
