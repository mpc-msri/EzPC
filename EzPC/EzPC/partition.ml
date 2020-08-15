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
open Codegenast
open Stdint

type gamma = {
    fdefs: (string * (qualifier list * binder list * stmt * ret_typ)) list;
    bindings: (var * (typ * string)) list
  }

let should_partition (g:gamma) (s:stmt) :bool =
  match s.data with
  | Assign (_, { data = App (f, _) }) | Decl (_, _, Some ({ data = App (f, _) })) ->
     List.assoc f g.fdefs |> (fun (quals, _, _, _) -> quals |> List.mem Partition)
  | Call (f, _) -> List.assoc f g.fdefs |> (fun (quals, _, _, _) -> quals |> List.mem Partition)
  | For (quals, _, _, _, _) -> quals |> List.mem Partition
  | _ -> false

type partition = stmt list

let partition_to_string (p:partition) :string =
  p |> List.fold_left (fun acc s -> acc ^ "; " ^ (stmt_to_string s)) ""

let split (g:gamma) (s:partition) :partition list =
  let rec split_current (s:partition) (acc:partition) :(partition * partition) =
    match s with
    | []     -> List.rev acc, []
    | hd::tl -> if should_partition g hd then List.rev acc, s else split_current tl (hd::acc)
  in
  let rec aux (current:partition) (acc:partition list) :partition list =
    let p1, p2 = split_current current [] in
    match p2 with
    | []     -> List.rev (p1::acc)
    | hd::tl -> aux tl ([hd]::p1::acc)
  in
  aux s []

(*
 * convert s to a list of basic, top-level stmts
 * we don't descend inside For and If, since partitioning is currently top-level only
 *)
let rec stmt_to_list (s:stmt) :stmt list =
  match s.data with
  | Seq (s1, s2) -> List.append (stmt_to_list s1) (stmt_to_list s2)
  | _ -> [s]

let rec list_to_stmt (l:stmt list) :stmt =
  match l with
  | []   -> Skip "" |> mk_dsyntax ""
  | s::l -> Seq (s, l |> list_to_stmt) |> mk_dsyntax ""
       
let rec codegen_stmt_list_to_codegen_stmt (l:codegen_stmt list) :codegen_stmt =
  match l with
  | []     -> failwith "Impossible, did not expect an empty list in codegen_stmt_list_to_codegen_stmt!"
  | [s]    -> s
  | hd::tl -> Seq_codegen (hd, tl |> codegen_stmt_list_to_codegen_stmt)
       
let get_fresh_filename :unit -> string =
  let r = ref 0 in
  fun _ -> let s = "__ezpc_shares_" ^ (string_of_int !r) in r := !r + 1; s

let gamma_to_string (g:gamma) :string = g.bindings |> List.fold_left (fun s (v,_) -> s ^ " " ^ v.name) ""
                                                            
let gamma_to_dump_stmts (g:gamma) :codegen_stmt list =
  let pre = Base_s (Skip ("Begin writing shares, variables in scope: " ^ (gamma_to_string g)) |> mk_dsyntax "") in
  let post = Base_s (Skip "End writing shares" |> mk_dsyntax "") in

  [pre]

  @ (g.bindings |> List.fold_left (fun l (x, (t, f)) ->
                       List.append l [ if Config.get_debug_partitions ()
                                       then Seq_codegen (Base_s (Skip ("Dumping " ^ x.name ^ " for debugging") |> mk_dsyntax ""),
                                                                 Base_s (Output (Role Both |> mk_dsyntax "",
                                                                                 Var x |> mk_dsyntax "",
                                                                                 Some t) |> mk_dsyntax ""))
                                       else Base_s (Skip "" |> mk_dsyntax "");
                                       Dump_interim (Var x |> mk_dsyntax "", t, f) ]
                     ) [])
  @ [post]
  
let gamma_to_read_stmts (g:gamma) :codegen_stmt list =
  let pre = Base_s (Skip ("Begin reading shares, variables in scope: " ^ (gamma_to_string g)) |> mk_dsyntax "") in
  let post = Base_s (Skip "End reading shares" |> mk_dsyntax "") in

  [pre]

  @ (g.bindings |> List.fold_left (fun l (x, (t, f)) ->
                       List.append l [ Read_interim (Var x |> mk_dsyntax "", t, f);

                                       if Config.get_debug_partitions ()
                                       then Seq_codegen (Base_s (Skip ("Dumping " ^ x.name ^ " for debugging") |> mk_dsyntax ""),
                                                         Base_s (Output (Role Both |> mk_dsyntax "",
                                                                         Var x |> mk_dsyntax "",
                                                                         Some t) |> mk_dsyntax ""))
                                       else Base_s (Skip "" |> mk_dsyntax "") ]
                     ) [])

  @ [post]

let gamma_to_decls (g:gamma) :codegen_stmt list =
  (List.rev g.bindings) |> List.fold_left (fun l (x, (t, _)) -> List.append l [(Base_s (Decl (t, Var x |> mk_dsyntax "", None) |> mk_dsyntax ""))]) []

let var_used_in_partition_list (pl:partition list) (x:var) :bool =
  pl |> List.exists (fun p -> p |> List.exists (fun s -> x |> var_used_in_stmt s))
                                                      
let get_output_gamma (g:gamma) (p:partition) (rest:partition list) :gamma =
  let g =
    p |> List.fold_left (fun g s ->
             match s.data with
             | Decl (t, e_var, _) when is_var e_var ->
                { g with bindings = List.append g.bindings [(get_var e_var, (t, get_fresh_filename ()))] }
             | Input (_, e_var, t) when is_var e_var ->
                { g with bindings = List.append g.bindings [(get_var e_var, (t, get_fresh_filename ()))] }
             | _ -> g) g
  in
  { g with bindings = g.bindings |> List.filter (fun (x, _) -> x |> var_used_in_partition_list rest) }

let add_partition_prefix_and_suffix (g:gamma) (p:partition) (rest:partition list) (is_first:bool) (is_last:bool) :gamma * (codegen_stmt list) =
  let og = get_output_gamma g p rest in
  let cp = p |> List.map (fun s -> Base_s s) in

  let rs = g |> gamma_to_read_stmts in
  let decls = g |> gamma_to_decls in
  let ds = og |> gamma_to_dump_stmts in

  og,
  if is_first then cp @ ds
  else if is_last then decls @ rs @ cp
  else decls @ rs @ cp @ ds

let rec check_no_decls (s:stmt) :bool =
  match s.data with
  | Decl _ -> false
  | Assign _ -> true
  | Call _ -> true
  | For _ -> true
  | While _ -> true
  | If_else _ -> true
  | Return _ -> true
  | Seq (s1, s2) -> check_no_decls s1 && check_no_decls s2
  | Input _ -> false
  | Output _ -> true
  | Skip _ -> true

let fresh_unroll_var :string -> var =
  let r = ref 0 in
  fun s -> r := !r + 1; { name = s ^ "__" ^ string_of_int !r; index = 0 }

let fresh_inline_var :string -> var =
  let r = ref 0 in
  fun s -> r := !r + 1; { name = s ^ "__" ^ string_of_int !r; index = 0 }

(*
 * Collect top-level declarations in this statement
 * Inlining functions and unrolling loops can introduce duplicate declarations,
 *   so the corresponding code needs to "freshen" then up
 *)           
let collect_decls (turn_into_assignment:bool) (s:stmt) :(typ * expr) list * stmt =
  let rec aux (acc:(typ * expr) list) (s:stmt) :(typ * expr) list * stmt =
    match s.data with
    | Decl (t, e, eopt) ->
       (t, e)::acc,
       if turn_into_assignment then
         { s with data = if eopt |> is_some then Assign (e, eopt |> get_opt) else Skip "" }
       else s
    | Seq (s1, s2) -> s1 |> aux acc |> (fun (acc, s1) -> s2 |> aux acc |> (fun (acc, s2) -> acc, { s with data = Seq (s1, s2) }))
    | _ -> acc, s
  in
  aux [] s

(*
 * Modify the function body with args so that it can be inlined
 *)
let inline_fn (bs:binder list) (body:stmt) (ret_t:ret_typ) (args:expr list) (rng:Global.Metadata.metadata) :stmt option =
  let ret_ok = match ret_t with | Void _ -> true | _ -> false in
  (*
   * We can inline if the arguments are simple expressions
   * Array reads for subarrays, since we don't allow aliasing on arrays
   *)
  let args_ok = args |> List.for_all (fun a -> match a.data with | Role _ | Const _ | Var _ | Array_read _ -> true | _ -> false)
  in
  (* This actually should never happen, tc should have checked it already *)
  if not ret_ok then None
  else if not args_ok then begin
      print_string ("WARNING: Unable to inline function call at " ^ (Global.Metadata.sprint_metadata "" rng) ^
                      ", as the arguments are non-atomic (constants, variables, and array reads)");
      None
    end
  else
    (* Inline, first freshen up the top-level declarations in the body *)
    let body =
      body |> collect_decls false |> fst |> List.map snd |>
        List.fold_left (fun body e_from ->
            body |> Ast.subst_stmt e_from (Var (fresh_inline_var (e_from |> get_var_name)) |> mk_dsyntax "")) body
    in
    Seq (Skip ("Begin inlining function call at location: " ^ Global.Metadata.sprint_metadata "" rng) |> mk_dsyntax "",
         Seq (List.fold_left2 (fun body (v, t) arg ->
                  match t.data with
                  | Base _ ->
                     (*
                      * Since scalar arguments are passed by value, create a copy
                      *)
                     let e_new = v.name |> fresh_inline_var |> (fun v -> Var v |> mk_dsyntax "") in
                     let comment = "Temporary variable for argument " ^ (expr_to_string arg) in
                     let decl = Decl (t, e_new, Some arg) |> mk_dsyntax comment in
                     Seq (decl, body |> Ast.subst_stmt (Var v |> mk_dsyntax "") e_new) |> mk_dsyntax ""
                  | _ -> body |> Ast.subst_stmt (Var v |> mk_dsyntax "") arg  (* Arrays are passed by reference *)
                ) body bs args,
              Skip ("End inlining function call at location: " ^ Global.Metadata.sprint_metadata "" rng) |> mk_dsyntax "") |> mk_dsyntax "") |> mk_dsyntax "" |> some

let rec inline_fn_calls (g:gamma) (s:stmt) :stmt =
  let rng = s.metadata in
  let aux (s:stmt') :stmt' =
    match s with
    | Decl _ -> s
    | Assign _ -> s
    | Call (f, args) ->
       let (quals, bs, body, ret_t) = g.fdefs |> List.assoc f in
       if quals |> List.mem Inline then
         (* Inline function calls in the function body itself first *)
         let body = body |> inline_fn_calls g in
         let body_opt = inline_fn bs body ret_t args rng in
         if body_opt |> is_some then body_opt |> get_opt |> (fun body -> body.data )
         else s
       else s
    | For (quals, e_index, e_from, e_to, body) -> For (quals, e_index, e_from, e_to, body |> inline_fn_calls g)
    | While (e, body) -> While (e, body |> inline_fn_calls g)
    | If_else (e, s_then, s_else_opt) -> If_else (e, s_then |> inline_fn_calls g, map_opt s_else_opt (inline_fn_calls g))
    | Return _ -> s
    | Seq (s1, s2) -> Seq (s1 |> inline_fn_calls g, s2 |> inline_fn_calls g)
    | Input _ -> s
    | Output _ -> s
    | Skip _ -> s
  in
  s.data |> aux |> (fun s' -> { s with data = s' })
  
let rec unroll_loops (s:stmt) :stmt =
  let rng = s.metadata in
  let aux (s:stmt') :stmt' =
    match s with
    | Decl _ -> s
    | Assign _ -> s
    | Call _ -> s
    | For (quals, e_var, { data = Const (Int32C n1) }, { data = Const (Int32C n2) }, body)
         when List.mem Unroll quals && Int32.compare n1 n2 < 0 ->
       let body = unroll_loops body in
       let body =
         body |> collect_decls false |> (fun (l, body) ->
           l |> List.fold_left (fun body (t, e_from) ->
                    let x = Var (fresh_unroll_var (e_from |> get_var_name)) |> mk_dsyntax "" in
                    body |> subst_stmt e_from x
                  ) body
         )
       in
       let body0 = body |> subst_stmt e_var (Const (Int32C n1) |> mk_dsyntax "") in
       let body = body |> collect_decls true |> snd in
       let rec aux (n:int32) (s:stmt) :stmt' =
         if Int32.compare n n2 >= 0 then s.data
         else aux (Int32.add n Int32.one)
                  (Seq (s, body |> subst_stmt e_var (Const (Int32C n) |> mk_dsyntax "")) |> mk_dsyntax "")
       in
       let unrolled = aux (Int32.add n1 Int32.one) body0 in
       let pre = Skip ("Begin unrolling loop at source location: " ^ (Global.Metadata.sprint_metadata "" rng)) |> mk_dsyntax ""
       in
       let post = Skip ("End unrolling loop at source location: " ^ (Global.Metadata.sprint_metadata "" rng)) |> mk_dsyntax ""
       in
       Seq (pre, Seq (unrolled |> mk_dsyntax "", post) |> mk_dsyntax "")
    | For (q, e_index, e_upper, e_lower, body) -> For (q, e_index, e_upper, e_lower, unroll_loops body)
    | While (e, s) -> While (e, unroll_loops s)
    | If_else (e, s_then, s_else_opt) -> If_else (e, unroll_loops s_then, map_opt s_else_opt unroll_loops)
    | Return _ -> s
    | Seq (s1, s2) -> Seq (unroll_loops s1, unroll_loops s2)
    | Input _ -> s
    | Output _ -> s
    | Skip _ -> s
  in
  s.data |> aux |> (fun s' -> { s with data = s' })

let partition (p:program) :codegen_program =
  let g = {
      fdefs = p |> List.map (fun d -> match d.data with
                             | Fun (quals, f, bs, body, ret_t) -> [f, (quals, bs, body, ret_t)]
                             | Extern_fun (quals, f, bs, ret_t) -> [f, (quals, bs, (Skip "Adding NoOp for extern func.") |> mk_dsyntax "", ret_t)]
                             | _ -> []) |> List.flatten;
      bindings = []
    }
  in

  let rest, main =
    let rest, main = p |> List.rev |> (fun p -> p |> List.tl |> List.rev, p |> List.hd) in
    match main.data with
    | Fun (_, _, _, main, _) -> rest, main
    | _ -> failwith "Impossible! last program declaration in partition has to be main"
  in

  if (Config.get_codegen () = Config.CPP || Config.get_codegen () = Config.CPPRING) && not (Config.get_debug_partitions ()) then rest, [Base_s main]
  else
    let main = main |> inline_fn_calls g in
    print_msg "Partition: inlined function calls ...";

    let main = main |> unroll_loops in
    print_msg "Partition: unrolled loops ...";

    let ps = main |> stmt_to_list |> split g |> List.filter (fun p -> not (p = [])) in

    let mains =  (* list of (codegen_stmt list), each (codegen_elmt list) is a main body *)
      match ps with
      | []     -> failwith "Impossible, cannot have empty partitions!"
      | [ p ]  -> [p |> List.map (fun s -> Base_s s)]
      | hd::tl ->
         let g, cs_hd = add_partition_prefix_and_suffix g hd tl true false in

         let rec aux (ps:partition list) (acc:gamma * ((codegen_stmt list) list)) :(codegen_stmt list) list =
           match ps with
           | []     -> acc |> snd |> List.rev
           | hd::tl ->
              let g, cs_hd = add_partition_prefix_and_suffix (acc |> fst) hd tl false (tl = []) in
              aux tl (g, (cs_hd::(acc |> snd)))
         in

         aux tl (g, [cs_hd])
    in

    let mains = mains |> List.map codegen_stmt_list_to_codegen_stmt in

    rest, mains
