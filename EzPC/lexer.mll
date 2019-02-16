{
open Parser
open Lexing
open Stdint
let incr_linenum lexbuf =
  let pos = lexbuf.Lexing.lex_curr_p in
  lexbuf.Lexing.lex_curr_p <-
    { pos with
      Lexing.pos_lnum = pos.Lexing.pos_lnum + 1;
      Lexing.pos_bol = pos.Lexing.pos_cnum;
    }

exception LexErr of string
exception ParseErr of string
exception Error of string

let error msg start finish  = 
    Printf.sprintf "(line %d: char %d..%d): %s" (start.pos_lnum -1)
          (start.pos_cnum -start.pos_bol) (finish.pos_cnum - finish.pos_bol) msg

let lex_error lexbuf = 
    raise ( LexErr (error (lexeme lexbuf) (lexeme_start_p lexbuf) (lexeme_end_p lexbuf)))

let cvt_int32_literal s =
  Int32.of_string s
let cvt_int64_literal s =
  Int64.of_string (String.sub s 0 (String.length s - 1))
let cvt_uint32_literal s =
  Uint32.of_string (String.sub s 0 (String.length s - 1))
let cvt_uint64_literal s =
  Uint64.of_string (String.sub s 0 (String.length s - 2))
}

let white = [' ' '\t']+
let digit = ['0'-'9']
let int = digit+
let letter = ['a'-'z' 'A'-'Z']
let all = ['a'-'z' 'A'-'Z' '0'-'9']
let id = letter all*
let bool = "true" | "false"
let eol = '\r' | '\n' | "\r\n"

rule read = 
  parse
  | white { read lexbuf }
  | eol { incr_linenum lexbuf; read lexbuf }
  | '(' '*' { nested_comment 0 lexbuf }
  | "(" { LPAREN }
  | ")" { RPAREN }
  | "{" { LBRACE }
  | "}" { RBRACE }
  | "[" { LBRACKET }
  | "]" { RBRACKET }
  | ";"   { SEMICOLON }
  | ","   { COMMA }
  | "?"   { QUESTION_MARK }
  | ":"   { COLON }
  | "="   { EQUALS }
  | "~"   { BITWISE_NEG }
  | "!"   { NOT }
  | "+"   { SUM }
  | "-"   { SUB }
  | "*"   { MUL }
  | "/"   { DIV }
  | "%"   { MOD }
  | "^^"  { POW }
  | ">>"  { R_SHIFT_A }
  | "<<"  { L_SHIFT }
  | "&"   { BITWISE_AND }
  | "|"   { BITWISE_OR }
  | "^"   { BITWISE_XOR }
  | "&&"  { AND }
  | "||"  { OR }
  | "@"   { XOR }
  | ">>>" { R_SHIFT_L }
  | "|>"  { SUBSUMPTION }
  | "<"   { LESS_THAN }
  | ">"   { GREATER_THAN }
  | ">="  { GREATER_THAN_EQUAL }
  | "<="  { LESS_THAN_EQUAL }
  | "=="  { IS_EQUAL }
  | "_"   { UNDERSCORE }
  | "int32" { TINT32 }
  | "int64" { TINT64 }
  | "uint32" { TUINT32 }
  | "uint64" { TUINT64 }
  | "bool"  { TBOOL }
  | "void" {TVOID }
  | "true" { TRUE }
  | "false" { FALSE }
  | "al" { ARITHMETIC }
  | "bl" { BOOLEAN }
  | "pl" { PUBLIC }
  | "output" { OUTPUT }
  | "input" { INPUT }
  | "return" { RETURN }
  | "partition" { PARTITION }
  | "inline" { INLINE }
  | "unroll" { UNROLL }
  | "const" { CONST }
  | "if" { IF }
  | "def" { DEF }
  | "else" { ELSE }
  | "for" { FOR }
  | "while" { WHILE }
  | "SERVER" { SERVER }
  | "CLIENT" { CLIENT }
  | "ALL" { ALL }
  | id    { ID (Lexing.lexeme lexbuf) }
  | int { try INT32 (cvt_int32_literal (Lexing.lexeme lexbuf))
          with Failure _ -> raise (Error ("literal overflow int32")) }
  | int "u" { try UINT32 (cvt_uint32_literal (Lexing.lexeme lexbuf))
          with Failure _ -> raise (Error ("literal overflow uint32")) }
  | int "L" { try INT64 (cvt_int64_literal (Lexing.lexeme lexbuf))
              with Failure _ -> raise (Error ("literal overflow int64")) }
  | int "uL" { try UINT64 (cvt_uint64_literal (Lexing.lexeme lexbuf))
              with Failure _ -> raise (Error ("literal overflow uint64")) }
  | eof   { EOF }
  | _ { lex_error lexbuf }


and nested_comment level = parse
  | '(' '*' { nested_comment (level + 1) lexbuf }
  | '*' ')' { if level = 0 then 
                read lexbuf
              else
                nested_comment (level - 1) lexbuf }
  | eol { incr_linenum lexbuf; nested_comment level lexbuf }
  | _ { nested_comment level lexbuf }
