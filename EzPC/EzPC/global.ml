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

module Metadata = struct
  type loc = Lexing.position

  and metadata =
    | Root of loc * loc * string
    | Stepped of metadata

  let rec sprint_metadata prefix = function
    | Stepped p -> prefix
    | Root (loc1, loc2, _) ->
       let print_loc loc =
         string_of_int (loc.Lexing.pos_lnum - 1) ^ "," ^ string_of_int (loc.Lexing.pos_cnum - loc.Lexing.pos_bol)
       in
       prefix ^ "(" ^ print_loc loc1 ^ "-" ^ print_loc loc2 ^ ")"
end

let dmeta (s:string) = Metadata.Root (Lexing.dummy_pos, Lexing.dummy_pos, s)

let ddmeta = Metadata.Root (Lexing.dummy_pos, Lexing.dummy_pos, "")

type range = Metadata.metadata

let get_comment (d:Metadata.metadata) :string =
  match d with
  | Metadata.Root (_, _, s) -> s
  | _ -> ""
