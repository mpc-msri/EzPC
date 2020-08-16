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

open Ast

(*
 * This is the AST over which codegen operates -- it is shared between the ABY and oblivc backends
 *
 * Basically a statement node in the source AST can result in multiple statements in the output code
 * For example, input of an array is a single statement in the source
 *              but results in a for loop in the output code
 * Instead of playing with strings, the codegen for Input will create AST nodes for the for loop etc.
 *   and then call codegen on them, which will output proper code
 *)

type codegen_expr =
  | Base_e              of expr
  | Input_g             of role * secret_label * var * base_type  (* base_type is used by the oblivc backend to output the correct format string specifier in the oblivc input function *)
  | Output_g            of role * secret_label * codegen_expr     (* only used by the ABY backend, in oblivc output is a statement, see Output_s below *)
  | Dummy_g             of secret_label * base_type
  | Clear_val           of codegen_expr * base_type
  | Conditional_codegen of codegen_expr * codegen_expr * codegen_expr * label
  | App_codegen_expr    of string * codegen_expr list

type codegen_stmt =
  | Base_s         of stmt
  | App_codegen    of string * codegen_expr list
  | Cin            of string * codegen_expr * base_type  (* base_type is used in the oblivc backend for scanf format string *)
  | Cout           of string * codegen_expr * base_type  (* base_type is used in the oblivc backend for printf format string *)
  | Dump_interim   of expr * typ * string
  | Read_interim   of expr * typ * string
  | Open_file      of bool * string * string             (* true if write, false for read; var name for stream; file *)
  | Close_file     of string                             (* stream name *)
  | Assign_codegen of codegen_expr * codegen_expr
  | For_codegen    of codegen_expr * codegen_expr * codegen_expr * codegen_stmt
  | If_codegen     of label * codegen_expr * codegen_stmt * codegen_stmt option  (* the label is used only in the oblivc backend, since secret conditionals become obliv if statements *)
  | Seq_codegen    of codegen_stmt * codegen_stmt
  | Output_s       of role * codegen_expr * codegen_expr * base_type (* only used by oblivc backend, first expression is add_to_output_queue return value -- this is where the clear output will be put, second expression is the secret expression, base_type is the base type of the secret expression *)

type codegen_program = global list * codegen_stmt list
