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
