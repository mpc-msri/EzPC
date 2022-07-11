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

open Stdint
open Printf
open Utils
open Config
open Ast
open Resolvevars
open Infer
open Tc
open Optimizer
open Partition
open Codegen
open Codegenoblivc
open Codegenemp

let load_file (f:string) :string =
  let ic = open_in f in
  let buf = Buffer.create 0 in

  let rec read (_:unit) :unit =
    Buffer.add_char buf '\n';
    try
      let s = input_line ic in
      Buffer.add_string buf s;
      read ()
    with
      _ -> ()
  in
  read ();
  Buffer.contents buf

let parse s = 
  let lexbuf = Lexing.from_string s in
  try Parser.program Lexer.read lexbuf
  with e -> 
      let cur_loc = lexbuf.Lexing.lex_curr_p in
      let mtd = Global.Metadata.Root (cur_loc, cur_loc, "") in
      print_string ((Global.Metadata.sprint_metadata "Error in parsing : " mtd) ^ "\n");
      raise e

let tc_and_codegen (p:program) (file:string) :unit =
  print_msg "Running the inference pass";
  let p = infer p in
  print_msg "Inferred binop labels and coercions";
  let x =
    bind (tc_program p) (fun r ->
           print_msg "Typechecked the program";
           let p =
             if Config.get_tac () then
               let p = p |> Optimizer.tac_program in
               print_msg "Applied 3-address code transformation";
               p
             else p
           in
           let p =
             if Config.get_cse () then
               let p = p |> Optimizer.cse_program in
               print_msg "Applied common subexpresion elimination optimization";
               p
             else p
           in
           let p =
             if Config.get_codegen () = CPP || Config.get_codegen () = CPPRING 
             then p |> erase_labels_program
             else p
           in
           let p = partition p in
           print_msg ("Split the program into " ^ string_of_int (p |> snd |> List.length) ^ " partition(s), generating .cpp files");
           if Config.get_codegen () = OBLIVC then Codegenoblivc.o_program p file
           else if Config.get_codegen () = PORTHOS then Codegenporthos.o_program p file
           else if Config.get_codegen () = SCI then Codegensci.o_program p file
           else if Config.get_codegen () = FSS then Codegenfss.o_program p file
           else if Config.get_codegen () = CPPRING then Codegencppring.o_program p file
           else if Config.get_codegen () = EMP then Codegenemp.o_program p file
           else Codegen.o_program p file;
           Well_typed ()) in
  match x with
  | Type_error (s, r) -> begin
      print_string ("Type_error" ^ (Global.Metadata.sprint_metadata "" r) ^ ": " ^ s ^ "\n");
      exit 1
    end
  | _ -> if Config.get_codegen () = OBLIVC then print_msg ("Output written to file " ^ file ^ "0.oc")
           else print_msg ("Output(s) written to file(s) " ^ file ^ "(n).cpp, n = 0, ...")

let help () =
  print_string "\n./ezpc <source file name> <output file name (prefix)> <optional bitlen, default = 32>\n";
  exit 0


(*****)

let o_prefix :string ref = ref ""
let input_file :string ref = ref ""

let specs = Arg.align [
                ("--bitlen", Arg.Int Config.set_bitlen, "Bitlength to be used for shares");
                ("--codegen", Arg.String (fun s -> match s with
                                                   | "ABY" -> ABY |> Config.set_codegen
                                                   | "CPP" -> CPP |> Config.set_codegen
                                                   | "OBLIVC" -> OBLIVC |> Config.set_codegen
                                                   | "PORTHOS" -> PORTHOS |> Config.set_codegen
                                                   | "SCI" -> SCI |> Config.set_codegen
                                                   | "FSS" -> FSS |> Config.set_codegen
                                                   | "CPPRING" -> CPPRING |> Config.set_codegen
                                                   | "EMP" -> EMP |> Config.set_codegen
                                                   | _ -> failwith "Invalid codegen mode"),
                 " Codegen mode (ABY or CPP or OBLIVC or PORTHOS or SCI or CPPRING or FSS or EMP, default ABY)");
                ("--o_prefix", Arg.String (fun s -> o_prefix := s), " Prefix for output files, default is the input file prefix");
                ("--disable-tac", Arg.Unit Config.disable_tac, " Disable 3-address code transformation (also disables the CSE optimization)");
                ("--disable-cse", Arg.Unit Config.disable_cse, " Disable Common Subexpression Elimination optimization");
                ("--dummy_inputs", Arg.Unit Config.set_dummy_inputs, " Use dummy inputs in the generated code (disabled by default)");
                ("--bool_sharing", Arg.String (fun s -> match s with
                                                        | "Yao" -> Yao |> Config.set_bool_sharing_mode
                                                        | "GMW" -> GMW |> Config.set_bool_sharing_mode
                                                        | _ -> failwith "Invalid bool sharing mode"),
                 " Bool sharing mode (Yao or GMW, default Yao)");
                ("--shares_dir", Arg.String (fun s -> Config.set_shares_dir s), " Directory where share files should be created");
                ("--debug_partitions", Arg.Unit Config.set_debug_partitions, " Debug partitions (if codegen is ABY then dump shares in clear, if codegen is CPP then generate partitions)");
                ("--modulo", Arg.String Config.set_modulo, 
                  "Modulo to be used for shares. Applicable for CPPRING/SCI backend. 
                  For SCI, for backend type OT, this should be power of 2 and for backend type HE, this should be a prime.");
                ("--backend", Arg.String (fun s -> match s with
                                                   | "OT" -> OT |> Config.set_sci_backend
                                                   | "HE" -> HE |> Config.set_sci_backend
                                                   | _ -> failwith "Invalid backend type"),
                 "SCI Backend Type (OT or HE, default OT).");
                ("--sf", Arg.Int Config.set_sf, "Scale factor to be used in compilation. Valid only for PORTHOS.");
                ("--l", Arg.Unit Config.set_libmode, "Dump library (should not contain main function, works only for FSS Mode)")
              ]
let _ =
  Random.self_init ();
  
  let _ = Arg.parse specs (fun f -> input_file := f) ("usage: ezpc [options] [input file]. options are:") in

  let _ =
    if Config.get_codegen () <> FSS && Config.get_libmode () 
    then failwith "ezpc: library mode is only supported for FSS backend";

    if Config.get_codegen () = CPP && Config.get_actual_bitlen () <> 32 && Config.get_actual_bitlen () <> 64
    then failwith "CPP codegen requires bitlen of 32/64.";

    if Config.get_codegen () <> PORTHOS && Config.get_sf () <> 0
    then failwith "sf only valid for PORTHOS.";

    if Config.get_codegen () <> CPPRING && Config.get_codegen () <> SCI && Config.get_codegen () <> FSS
      && (Config.get_modulo () <> (Uint64.shift_left (Uint64.of_int 1) (Config.get_bitlen ())) || (Config.get_bitlen () <> 32 && Config.get_bitlen () <> 64)) 
    then failwith "Modulo and {bitlen not equal to 32/64} only supported for CPPRING/SCI/FSS backend.";

    if Config.get_codegen () = CPPRING && (Config.get_bitlen () = 64 || Config.get_bitlen () = 32) 
      && Config.get_modulo () = (Uint64.shift_left (Uint64.of_int 1) (Config.get_bitlen ()))
    then begin
    print_msg ("CPPRING codegen called for 1<<{32/64} ring. Switching to codegen CPP with bitlen = 32/64.");
    Config.set_codegen CPP
    end
  in

  let input_file_name = !input_file in
  let b = Str.string_match (Str.regexp "\\(.*\\)\\.ezpc") input_file_name 0 in
  if not b then failwith "Invalid input file name (it must end with .ezpc)";
  let prefix = Str.matched_group 1 input_file_name in
  
  if !o_prefix = "" then o_prefix := prefix;

  let _ = 
    if Config.get_codegen () = SCI then begin
      let backend_type = if Config.get_sci_backend () = OT then "OT" else "HE" in
      o_prefix := !o_prefix ^ "_" ^ backend_type;
    end
  in

  let file_contents = load_file !input_file in
  let file_contents = if Config.get_libmode() then (file_contents ^ "\ndef void main() {}\n") else file_contents in
  print_msg ("Read file " ^ !input_file);
  let gen_ast = parse file_contents in
  print_msg ("Parsed file " ^ !input_file);
  match resolve_vars_program gen_ast with
  | Binding_error (s, r) -> begin
      print_string ("Binding_error" ^ (Global.Metadata.sprint_metadata "" r) ^ ": " ^ s);
      exit 1
    end
  | Success p -> tc_and_codegen p !o_prefix
