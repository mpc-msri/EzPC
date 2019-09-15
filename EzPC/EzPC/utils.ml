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
  

