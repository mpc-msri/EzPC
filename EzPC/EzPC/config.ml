type codegen =
  | ABY
  | CPP
  | OBLIVC
  | PORTHOS

type bool_sharing_mode =
  | Yao
  | GMW
  
type configuration = {
    bitlen: int;
    out_mode:codegen;
    tac: bool;
    cse: bool;
    dummy_inputs: bool;
    bool_sharing: bool_sharing_mode;
    shares_dir: string;
    debug_partitions: bool;
  }

let c_private :configuration ref = ref {
                                       bitlen = 32;
                                       out_mode = ABY;
                                       tac = true;
                                       cse = true;
                                       dummy_inputs = false;
                                       bool_sharing = Yao;
                                       shares_dir = "";
                                       debug_partitions = false;}

let set_bitlen (bitlen:int) :unit = c_private := { !c_private with bitlen = bitlen }

let get_bitlen () :int = !c_private.bitlen

let set_codegen (g:codegen) :unit = c_private := { !c_private with out_mode = g }

let get_codegen () :codegen = !c_private.out_mode

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
