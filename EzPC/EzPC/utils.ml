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

let is_none (x:'a option) :bool =
  match x with
  | Some _ -> false
  | None   -> true

let is_some (x:'a option) :bool = not (is_none x)
            
let get_opt (x:'a option) :'a =
  match x with
  | Some y -> y
  | _      -> failwith "get_opt: got none"

let some (x:'a) :'a option = Some x

let double_opt (x:('a option) option) :'a option = match x with
  | None -> None
  | Some x -> x
            
let map_opt (x:'a option) (f:'a -> 'b) :'b option =
  match x with
  | None -> None
  | Some x -> Some (f x)

type 'a stack = 'a list

let empty_stack :'a stack = []
              
let push_stack (s:'a stack) (x:'a) :'a stack = x::s

let pop_stack (s:'a stack) :('a * 'a stack) =
  match s with
  | []     -> failwith "Cannot pop an empty stack"
  | hd::tl -> hd, tl

let peek_stack (s:'a stack) :'a =
  match s with
  | []    -> failwith "Cannot peek in an empty stack"
  | hd::_ -> hd

let is_empty_stack (s:'a stack) :bool = s = []

let singleton_stack (x:'a) :'a stack = [x]

let print_msg (s:string) :unit = print_string (s ^ " ..."); print_newline (); flush_all ()
  

