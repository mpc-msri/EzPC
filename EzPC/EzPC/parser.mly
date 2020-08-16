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

%{
open Ast
open Lexing
open Stdint
exception LexErr of string
exception ParseErr of string

let error msg start finish = 
    Printf.sprintf "(line %d: char %d..%d): %s" start.pos_lnum 
          (start.pos_cnum -start.pos_bol) (finish.pos_cnum - finish.pos_bol) msg

let parse_error msg startterm endterm =
  raise (ParseErr (error msg (startterm) (endterm)))

let astnd thing start_pos end_pos = 
  { 
    Ast.data = thing ;
    Ast.metadata = Global.Metadata.Root (start_pos, end_pos, "") ;
  }

let rec mk_array_typ quals bt el start_pos end_pos =
  let data =
    match el with
    | [] -> bt
    | hd::tl -> Array (quals, mk_array_typ [] bt tl start_pos end_pos, hd)
  in
  {
    Ast.data = data;
    Ast.metadata = Global.Metadata.Root (start_pos, end_pos, "") ;
  }

let match_stmt_option msg str = 
  match str with
  | None -> (mk_dsyntax msg (Skip msg))
  | Some s -> s

%}

%token <Stdint.uint32> UINT32
%token <Stdint.uint64> UINT64
%token <Stdint.int32> INT32
%token <Stdint.int64> INT64
%token LPAREN RPAREN
%token LBRACE RBRACE
%token LBRACKET RBRACKET
%token SEMICOLON
%token OUTPUT INPUT
%token RETURN
%token COMMA
%token TINT32 TINT64 TUINT32 TUINT64
%token TBOOL TRUE FALSE
%token TVOID
%token ARITHMETIC BOOLEAN PUBLIC
%token IF ELSE
%token QUESTION_MARK COLON
%token UNDERSCORE
%token EQUALS
%token BITWISE_NEG NOT
%token SUM SUB MUL DIV MOD POW R_SHIFT_A L_SHIFT BITWISE_AND BITWISE_OR BITWISE_XOR AND OR XOR R_SHIFT_L
%token LESS_THAN GREATER_THAN IS_EQUAL GREATER_THAN_EQUAL LESS_THAN_EQUAL
%token SUBSUMPTION FOR WHILE
%token EOF
%token DEF
%token SERVER CLIENT ALL
%token PARTITION INLINE UNROLL CONST EXTERN
%token <string> ID

%nonassoc SEMICOLON
%nonassoc SEQ
%nonassoc CONDITIONAL
%nonassoc LABELLED_CONDITIONAL
%nonassoc THEN
%nonassoc ELSE
%nonassoc QUESTION_MARK
%left OR
%left XOR
%left AND
%left BITWISE_OR
%left BITWISE_XOR
%left BITWISE_AND
%left IS_EQUAL
%left GREATER_THAN GREATER_THAN_EQUAL LESS_THAN LESS_THAN_EQUAL
%left R_SHIFT_A L_SHIFT R_SHIFT_L
%left SUM SUB
%left MUL DIV MOD
%left POW
%nonassoc BINOP_ASSOC
%nonassoc BINOP_SOME_ASSOC
%nonassoc UNOP_ASSOC
%nonassoc UNOP_SOME_ASSOC
%nonassoc LBRACKET

%start program
%type <Ast.program> program
%type <Ast.global> global
%type <Ast.stmt> stmt

%%
   
program:
  | fl = nonempty_list(global); EOF { fl }
  ;

role:
  | SERVER { Ast.Server }
  | CLIENT { Ast.Client }
  | ALL { Ast.Both }
  ;

base_type:
  | TUINT32 { Ast.UInt32 }
  | TUINT64 { Ast.UInt64 }
  | TINT32 { Ast.Int32 }
  | TINT64 { Ast.Int64 }
  | TBOOL { Ast.Bool }
  ;

index_expr:
  | LBRACKET; e = expr; RBRACKET { e }

base_type_:
  | bt = base_type; { Ast.Base(bt, None) }
  | bt = base_type; UNDERSCORE; l = label { Ast.Base(bt, Some(l)) }

type_:
  | bt = base_type_ { astnd bt $startpos $endpos }
  | bt = base_type_; el = nonempty_list (index_expr) { mk_array_typ [] bt el $startpos $endpos }
  | CONST; bt = base_type_; el = nonempty_list (index_expr) { mk_array_typ [Immutable] bt el $startpos $endpos }
  ;

ret_typ_:
  | t = type_ { Ast.Typ t }
  | TVOID { Ast.Void (astnd () $startpos $endpos) }

binder:
  | typ = type_; var = ID { let v = { name = var; index = 0 } in (v, typ) }
  ;

binder_l:
  vl = separated_list(COMMA, binder)         { vl } ;
  ;

global:
  | PARTITION;  DEF; typ = ret_typ_; fn_name = ID; LPAREN; bl = binder_l; RPAREN; LBRACE; s = option(stmt); RBRACE { astnd(Fun ([Ast.Partition], fn_name, bl, (match_stmt_option "Empty Function" s), typ)) $startpos $endpos }
  | INLINE;  DEF; typ = ret_typ_; fn_name = ID; LPAREN; bl = binder_l; RPAREN; LBRACE; s = option(stmt); RBRACE { astnd(Fun ([Ast.Inline], fn_name, bl, (match_stmt_option "Empty Function" s), typ)) $startpos $endpos }
  | EXTERN; typ = ret_typ_; fn_name = ID; LPAREN; bl = binder_l; RPAREN; SEMICOLON { astnd(Extern_fun ([Ast.Extern], fn_name, bl, typ)) $startpos $endpos }
  | DEF; typ = ret_typ_; fn_name = ID; LPAREN; bl = binder_l; RPAREN; LBRACE; s = option(stmt); RBRACE { astnd(Fun ([], fn_name, bl, (match_stmt_option "Empty Function" s), typ)) $startpos $endpos }
  | typ = base_type; e = expr; EQUALS; value = expr; SEMICOLON { astnd (Global_const (astnd (Ast.Base (typ, Some Public)) $startpos $endpos, e, value)) $startpos $endpos }
  ;

(* This is ugly, we should write it better so that we don't have to write a rule for every qualifier *)
stmt:
  | typ = type_; e = expr { astnd (Ast.Decl(typ, e, None)) $startpos $endpos }
  | typ = type_; e = expr; EQUALS; value = expr { astnd (Ast.Decl(typ, e, Some(value))) $startpos $endpos }
  | e1 = expr; EQUALS; e2 = expr { astnd (Ast.Assign(e1,e2)) $startpos $endpos }
  | fn_name = ID; LPAREN; e = expr_l; RPAREN { astnd (Ast.Call (fn_name, e)) $startpos $endpos }
  | FOR; var = expr; EQUALS; LBRACKET; start = expr; COLON; last = expr; RBRACKET; LBRACE; s = option(stmt); RBRACE { astnd (Ast.For([], var, start, last, (match_stmt_option "Empty loop" s))) $startpos $endpos }
  | UNROLL; FOR; var = expr; EQUALS; LBRACKET; start = expr; COLON; last = expr; RBRACKET; LBRACE; s = stmt; RBRACE { astnd (Ast.For([Ast.Unroll], var, start, last, s)) $startpos $endpos }
  | PARTITION; FOR; var = expr; EQUALS; LBRACKET; start = expr; COLON; last = expr; RBRACKET; LBRACE; s = stmt; RBRACE { astnd (Ast.For([Ast.Partition], var, start, last, s)) $startpos $endpos }
  | WHILE; e = expr; LBRACE; s = option(stmt); RBRACE { astnd (Ast.While (e, (match_stmt_option "Empty loop" s))) $startpos $endpos }
  | IF; LPAREN; e = expr; RPAREN; s1 = stmt { astnd (Ast.If_else(e, s1, None)) $startpos $endpos }            %prec THEN
  | IF; LPAREN; e = expr; RPAREN; s1 = stmt; ELSE; s2 = stmt { astnd (Ast.If_else(e, s1, Some(s2))) $startpos $endpos }
  | RETURN; e = expr { astnd (Ast.Return (Some e)) $startpos $endpos }
  | RETURN { astnd (Ast.Return None) $startpos $endpos }
  | s1 = stmt; SEMICOLON; s2 = stmt { astnd (Ast.Seq(s1, s2)) $startpos $endpos } %prec SEQ
  | INPUT; LPAREN; party = expr; COMMA; var = expr; COMMA; typ = type_; RPAREN { astnd (Ast.Input(party, var, typ)) $startpos $endpos }
  | OUTPUT; LPAREN; party = expr; COMMA; var = expr; RPAREN { astnd (Ast.Output(party, var, None)) $startpos $endpos }
  | s = stmt_ { s }
  | error { parse_error "Stmt: " $startpos $endpos }
  ;

stmt_:
  | LBRACE; s = stmt; RBRACE  { s }
  | s = stmt; SEMICOLON { s }
  ;

expr:
  | r = role { astnd (Ast.Role r) $startpos $endpos }
  | i = const; { astnd (Ast.Const i) $startpos $endpos }
  | x = ID { let v = { name = x; index = 0; } in astnd (Ast.Var v) $startpos $endpos }
  | u = unop; e1 = expr; { astnd (Ast.Unop (u, e1, None)) $startpos $endpos }    %prec UNOP_ASSOC
  | u = unop; UNDERSCORE; l = label; e1 = expr; { astnd (Ast.Unop (u, e1, Some l)) $startpos $endpos }    %prec UNOP_SOME_ASSOC
  | e1 = expr; b = binop; e2 = expr; { astnd (Ast.Binop(b,e1,e2,None)) $startpos $endpos }    %prec BINOP_ASSOC
  | e1 = expr; b = binop; UNDERSCORE; l = label; e2 = expr; { astnd (Ast.Binop(b,e1,e2,Some(l))) $startpos $endpos }    %prec BINOP_SOME_ASSOC
  | e1 = expr; QUESTION_MARK; e2 = expr; COLON; e3 = expr { astnd (Ast.Conditional(e1,e2,e3,None)) $startpos $endpos }  %prec CONDITIONAL
  | e1 = expr; QUESTION_MARK; UNDERSCORE; l = label; e2 = expr; COLON; e3 = expr { astnd (Ast.Conditional(e1,e2,e3,Some(l))) $startpos $endpos }  %prec LABELLED_CONDITIONAL
  | var = expr; LBRACKET; idx = expr; RBRACKET { astnd (Ast.Array_read(var, idx)) $startpos $endpos }
  | fn_name = ID; LPAREN; e = expr_l; RPAREN { astnd (Ast.App(fn_name, e)) $startpos $endpos }
  | LESS_THAN; l1 = label; SUBSUMPTION; l2 = label; GREATER_THAN; e = expr { astnd (Ast.Subsumption(e, l1, l2)) $startpos $endpos }
  | e = expr_ { e }
  | error { parse_error "Expr: " $startpos $endpos}
  ;

expr_:
  | LPAREN; e = expr; RPAREN  { e }
  ;

expr_l:
  vl = separated_list(COMMA, expr)         { vl } ;
  ;

unop:
  | BITWISE_NEG { Ast.Bitwise_neg }
  | NOT { Ast.Not }

binop:
  | SUM { Ast.Sum }
  | SUB { Ast.Sub }
  | MUL { Ast.Mul }
  | DIV { Ast.Div }
  | MOD { Ast.Mod }
  | POW { Ast.Pow }
  | GREATER_THAN { Ast.Greater_than }
  | IS_EQUAL { Ast.Is_equal }
  | GREATER_THAN_EQUAL { Ast.Greater_than_equal }
  | LESS_THAN { Ast.Less_than }
  | LESS_THAN_EQUAL { Ast.Less_than_equal }
  | R_SHIFT_A { Ast.R_shift_a }
  | L_SHIFT { Ast.L_shift }
  | BITWISE_AND { Ast.Bitwise_and }
  | BITWISE_OR { Ast.Bitwise_or }
  | BITWISE_XOR { Ast.Bitwise_xor }
  | AND { Ast.And }
  | OR { Ast.Or }
  | XOR { Ast.Xor }
  | R_SHIFT_L { Ast.R_shift_l }
  ;

const:
  | i = INT32 { Ast.Int32C i }
  | i = UINT32 { Ast.UInt32C i }
  | SUB; i = INT32; { Ast.Int32C (Int32.neg i) }
  | SUB; i = UINT32; { Ast.UInt32C (Uint32.neg i) }
  | i = INT64 { Ast.Int64C i }
  | i = UINT64 { Ast.UInt64C i }
  | SUB; i = INT64; { Ast.Int64C (Int64.neg i) }
  | SUB; i = UINT64; { Ast.UInt64C (Uint64.neg i) }
  | TRUE { Ast.BoolC true }
  | FALSE { Ast.BoolC false }
  ;

label:
  | PUBLIC { Ast.Public }
  | i = secret_label { Ast.Secret i }
  ;

secret_label:
  | ARITHMETIC { Ast.Arithmetic }
  | BOOLEAN { Ast.Boolean }
  ;
