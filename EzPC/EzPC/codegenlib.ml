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

(* we will keep building the output in a buffer *)
type buffer = Buffer.t

(* buffers are modified in place *)
type 'a m = buffer -> 'a

type comp = unit m

let o_null :comp = fun _ -> ()
          
let o_uint32 (n:uint32) :comp = fun buf -> Buffer.add_string buf (Uint32.to_string n)

let o_uint64 (n:uint64) :comp = fun buf -> Buffer.add_string buf (Uint64.to_string n)

let o_int32 (n:int32) :comp = fun buf -> Buffer.add_string buf (Int32.to_string n)
                                                                                  
let o_int64 (n:int64) :comp = fun buf -> Buffer.add_string buf (Int64.to_string n)

let o_bool (b:bool) :comp = o_uint32 (if b then Uint32.of_int 1 else Uint32.of_int 0)

let o_str (s:string) :comp = fun buf -> Buffer.add_string buf s

let seq (f:comp) (g:comp) :comp = fun buf -> f buf; g buf

let o_space :comp = o_str " "

let o_newline :comp = o_str "\n"

let o_string_literal (s:string) :comp = seq (o_str "\"") (seq (o_str s) (o_str "\""))

let o_paren (c:comp) :comp = seq (o_str "(") (seq c (o_str ")"))

let o_with_semicolon (c:comp) :comp = seq c (o_str ";")
