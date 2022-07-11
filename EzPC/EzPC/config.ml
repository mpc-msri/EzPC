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

type codegen =
  | ABY
  | CPP
  | OBLIVC
  | PORTHOS
  | SCI
  | FSS
  | CPPRING
  | EMP

type bool_sharing_mode =
  | Yao
  | GMW

type sci_backend_type = 
  | OT
  | HE
  
type configuration = {
    bitlen: int;
    out_mode:codegen;
    tac: bool;
    cse: bool;
    dummy_inputs: bool;
    bool_sharing: bool_sharing_mode;
    shares_dir: string;
    debug_partitions: bool;
    actual_bitlen: int;
    modulo: uint64;
    sci_backend: sci_backend_type; 
    sf: int;
    libmode: bool;
  }

let c_private :configuration ref = ref {
                                       bitlen = 32;
                                       out_mode = ABY;
                                       tac = true;
                                       cse = true;
                                       dummy_inputs = false;
                                       bool_sharing = Yao;
                                       shares_dir = "";
                                       debug_partitions = false;
                                       actual_bitlen = 32;
                                       modulo = Uint64.of_int 0;
                                       sci_backend = OT;
                                       sf = 0;
                                       libmode = false;
                                       }

let set_bitlen (bitlen:int) :unit = 
  if (bitlen > 32) then 
    c_private := { !c_private with bitlen = 64 ; actual_bitlen = bitlen ; modulo = (Uint64.shift_left (Uint64.of_int 1) bitlen) }
  else 
    c_private := { !c_private with bitlen = 32 ; actual_bitlen = bitlen ; modulo = (Uint64.shift_left (Uint64.of_int 1) bitlen) }

let get_bitlen () :int = !c_private.bitlen

let get_actual_bitlen () :int = !c_private.actual_bitlen

let set_modulo (m:string) :unit = c_private := { !c_private with modulo = (Uint64.of_string m) }

let get_modulo () :uint64 = !c_private.modulo

let set_codegen (g:codegen) :unit = c_private := { !c_private with out_mode = g }

let get_codegen () :codegen = !c_private.out_mode

let set_sci_backend (b:sci_backend_type) :unit = c_private := { !c_private with sci_backend = b }

let get_sci_backend () :sci_backend_type = !c_private.sci_backend

let set_sf (m:int) :unit = c_private := { !c_private with sf = m }

let get_sf () :int = !c_private.sf

let disable_tac () :unit = c_private := { !c_private with tac = false; cse = false }

let get_tac () :bool = !c_private.tac

let disable_cse () :unit = c_private := { !c_private with cse = false }

let get_cse () :bool = !c_private.cse

let set_dummy_inputs () :unit = c_private := { !c_private with dummy_inputs = true }

let get_dummy_inputs () :bool = !c_private.dummy_inputs

let set_bool_sharing_mode (m:bool_sharing_mode) :unit = c_private := { !c_private with bool_sharing = m }

let get_bool_sharing_mode () :bool_sharing_mode = !c_private.bool_sharing

let set_shares_dir (s:string) :unit = c_private := { !c_private with shares_dir = s }

let get_shares_dir () :string = !c_private.shares_dir

let set_debug_partitions () :unit = c_private := { !c_private with debug_partitions = true }

let get_debug_partitions () :bool = !c_private.debug_partitions

let set_libmode () :unit = c_private := { !c_private with libmode = true }

let get_libmode () :bool = !c_private.libmode
