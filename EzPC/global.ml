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
