from torch import nn as nn

class Layer :
    def __init__(self, in_features, out_features, layer_no) :
        assert 0 < layer_no
        assert 0 < in_features
        assert 0 < out_features

        self.in_features = in_features
        self.out_features = out_features
        self.layer_no = layer_no

    def __str__(self) :
        return f"Layer({in_features}, {out_features})"
        
    def get_dims(self) :
        return (self.in_features, self.out_features)
    
    def get_wt_type(self, transpose=False) :
        dim1, dim2 = self.get_dims()
        if transpose : dim1, dim2 = dim2, dim1
        return f"float_fl[{dim2}][{dim1}]"
    
    def get_bias_type(self) :
        _, outf = self.get_dims()
        return f"float_fl[{outf}]"
    
    def get_wt_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}W, {self.get_wt_type()}) ; \n"
    
    def get_bias_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}b, {self.get_bias_type()}) ; \n"

        
class Network :
    def __init__(self, in_dim, hidden_dims, no_out) :
        assert in_dim > 0
        
        self.in_dim = in_dim
        self.no_out = no_out
        all_dims = [in_dim] + hidden_dims + [no_out]
        self.layers = []
        for ind, (n1, n2) in enumerate(zip(all_dims[:-1], all_dims[1:])) :
            self.layers += [Layer(n1, n2, ind+1)]
            
    @property
    def no_class(self) :
        return self.no_out
        
    def __len__(self) :
        return len(self.layers)
        
    def __getitem__(self, n) :
        return self.layers[n]
    
    def __str__(self) :
        str1 = "Neural network of the following layers - \n"
        for l in self.layers :
            str1 += f"\t{l}\n"
        return str1
    
class BeaconTranslator :
    def __init__(self, net, batch, iters, lr, loss="CE", momentum=False, name="") :
        self.batch = batch
        self.iters = iters
        self.net = net
        self.lr = lr
        self.loss = loss
        self.momentum = momentum
        self.name = name
        
    def __str__(self) :
        str1 = f"Using a batch size of {self.batch} to train for {self.iters} iterations with lr={self.lr}\n"
        str1 += str(net)
        return str1
    
    def get_batch_decl(self) :
        str1 = f"int32 BATCH={self.batch} ;\n"
        return str1
        
    def get_forward_header(self) :
        header_list = ""
        net_len = len(self.net)
        for ind, l in enumerate(self.net.layers) :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            header_list += f"{wt_type} layer{ind+1}W, {bias_type} layer{ind+1}b, "
            
        header_list += f"float_fl[BATCH][{self.net.in_dim}] layer1In, "
        for ind, l in enumerate(self.net.layers[:-1]) :
            _, outf = l.get_dims()
            header_list += f"\
bool_bl[BATCH][{outf}] layer{ind+1}ReluHot, \
float_fl[BATCH][{outf}] layer{ind+1}Out, \
float_fl[BATCH][{outf}] layer{ind+2}In, "
            
        header_list += f"float_fl[BATCH][{self.net.no_class}] fwdOut"        
        header = f"def void forward({header_list})"
        return header
    
    def get_forward_body(self) :
        net_len = len(self.net)
        body = ""
        for ind, l in enumerate(self.net.layers[:-1]) :
            inf, outf = l.get_dims()
            body += f"\
{l.get_wt_type(transpose=True)} layer{ind+1}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{ind+1}Temp ;\n\
Transpose({inf}, {outf}, layer{ind+1}W, layer{ind+1}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{ind+1}In, layer{ind+1}WReshaped, layer{ind+1}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{ind+1}Temp, layer{ind+1}b, layer{ind+1}Out) ;\n\
Relu2(BATCH, {outf}, layer{ind+1}Out, layer{ind+2}In, layer{ind+1}ReluHot) ;\n\
\n"
         
        ind = net_len
        l = self.net.layers[-1]
        inf, outf = l.get_dims()
        output_line_args = f"BATCH, {outf}, layer{ind}Temp, fwdOut"
        output_line = f"Softmax2({output_line_args}) ;\n" if self.loss == "CE" else f"Reassign2({self.batch}, {self.net.no_out}, layer{ind}Temp, fwdOut) ;\n"
        body += f"\
{l.get_wt_type(transpose=True)} layer{ind}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{ind}Temp ;\n\
Transpose({inf}, {outf}, layer{ind}W, layer{ind}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{ind}In, layer{ind}WReshaped, layer{ind}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{ind}Temp, layer{ind}b, layer{ind}Temp) ;\n\
{output_line}"
        
        return body
    
    def get_forward_func(self) :
        brace_open = '{'
        brace_close = '}'
        return f"{self.get_forward_header()} {brace_open}\n\
{self.get_forward_body()}\n\
{brace_close}"
    
    def get_backward_header(self) :
        header_list = f"float_fl[BATCH][{self.net.no_class}] target, float_fl[BATCH][{self.net.no_class}] fwdOut"
        
        net_len = len(self.net)
        for ind, l in enumerate(self.net.layers) :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            header_list += f", {wt_type} layer{ind+1}W, {bias_type} layer{ind+1}b"
            
        header_list += f", float_fl[BATCH][{self.net.in_dim}] layer1In"
        for ind, l in enumerate(self.net.layers[:-1]) :
            _, outf = l.get_dims()
            header_list += f"\
, bool_bl[BATCH][{outf}] layer{ind+1}ReluHot, \
float_fl[BATCH][{outf}] layer{ind+1}Out, \
float_fl[BATCH][{outf}] layer{ind+2}In"

        if self.momentum :
            for ind, l in enumerate(self.net.layers) :
                wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
                header_list += f", {wt_type} layer{ind+1}WMom, {bias_type} layer{ind+1}bMom"
              
        header = f"def void backward({header_list})"
        return header
    
    def get_backward_body(self) :
        net_len = len(self.net)
        backward_body = ""
        for i in range(net_len-1, -1, -1) :
            l = self.net.layers[i]
            inf, outf = l.get_dims()
            ind = i+1
            
            actDer_decl = f"float_fl[BATCH][{inf}] layer{ind-1}ActDer ;\n" if ind>1 else ''
            layer_decls = f"\
float_fl[BATCH][{outf}] layer{ind}Der ;\n\
float_fl[{inf}][BATCH] layer{ind}InReshaped ;\n\
float_fl[{inf}][{outf}] layer{ind}WDerReshaped ;\n\
float_fl[{outf}] layer{ind}bDer ;\n\
{actDer_decl}"
            
            arg1 = "BATCH"
            arg2 = outf
            arg3 = "fwdOut" if ind == net_len else f"layer{ind}ActDer"
            arg4 = "target" if ind == net_len else f"layer{ind}ReluHot"
            arg5 = f"layer{ind}Der"
            arg_list = f"{arg1}, {arg2}, {arg3}, {arg4}, {arg5}"

            arg_list = arg_list + (", true" if ind != net_len else '')
            func_name = "getOutDer" if ind == net_len else "IfElse2"
            
            act_call = f"{func_name}({arg_list}) ;\n"
            actDer_call = f"MatMul(BATCH, {outf}, {inf}, layer{ind}Der, layer{ind}W, layer{ind-1}ActDer) ;\n" if ind > 1 else ''
            layer_calls = f"\
{act_call}\
Transpose({inf}, BATCH, layer{ind}In, layer{ind}InReshaped) ;\n\
MatMul({inf}, BATCH, {outf}, layer{ind}InReshaped, layer{ind}Der, layer{ind}WDerReshaped) ;\n\
getBiasDer(BATCH, {outf}, layer{ind}Der, layer{ind}bDer) ;\n\
{actDer_call}"
            
            layer_body = f"{layer_decls}{layer_calls}\n"
            backward_body += layer_body
            
        trans_decls, trans_calls, update_calls = "", "", ""
        for ind, l in enumerate(self.net.layers) :
            inf, outf = l.get_dims()
            trans_decls += f"float_fl[{outf}][{inf}] layer{ind+1}WDer ; \n"
            trans_calls += f"Transpose({outf}, {inf}, layer{ind+1}WDerReshaped, layer{ind+1}WDer) ;\n"
            update_calls += f"\
updateWeights{'Momentum' if self.momentum else ''}2({outf}, {inf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind+1}W, layer{ind+1}WDer{', layer'+str(ind+1)+'WMom' if self.momentum else ''}) ;\n\
updateWeights{'Momentum' if self.momentum else ''}({outf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind+1}b, layer{ind+1}bDer{', layer'+str(ind+1)+'bMom' if self.momentum else ''}) ;\n"
            
        update_body = f"{trans_decls}{trans_calls}\n{update_calls}"
            
        return backward_body + update_body
            
    def get_backward_func(self) :
        brace_open, brace_close = '{', '}'
        return f"{self.get_backward_header()} {brace_open}\n\
{self.get_backward_body()}\n\
{brace_close}"
        
    def get_inputs(self) :
        str_inp = f"input(CLIENT, inp, float_fl[BATCH][{self.net.in_dim}]) ;\n"
        str_lab = f"input(CLIENT, target, float_fl[BATCH][{self.net.no_class}]) ;\n"
        str_ret = str_inp + str_lab
        
        for l in self.net.layers :
            str_ret += l.get_wt_input_stmt()
            str_ret += l.get_bias_input_stmt()
            
        return str_ret

    def get_mom_decls(self) :
        mom_str = ""
        for ind, l in enumerate(self.net.layers) :
            mom_str += f"\
float_fl[{l.out_features}][{l.in_features}] layer{ind+1}WMom ;\n\
float_fl[{l.out_features}] layer{ind+1}bMom ;\n\
"
        return mom_str
    
    def get_intermediate_decls(self) :
        decl_str = ""
        for ind, l in enumerate(self.net.layers[:-1]) :
            decl_str += f"\
    bool_bl[BATCH][{l.out_features}] layer{ind+1}ReluHot ;\n\
    float_fl[BATCH][{l.out_features}] layer{ind+1}Out ;\n\
    float_fl[BATCH][{l.out_features}] layer{ind+2}In ;\n\n"
            
        l = self.net.layers[-1]
        decl_str += f"\
    float_fl[BATCH][{self.net.no_class}] fwdOut ;\n\
    float_fl[1] loss ;\n"
        
        return decl_str
    
    def get_forward_call(self) :
        net_len = len(self.net)
        arg_list = ""
        for i in range(1, net_len+1) :
            arg_list += f"layer{i}W, layer{i}b, "
            
        arg_list += "inp, "
        for i in range(1, net_len) :
            arg_list += f"layer{i}ReluHot, layer{i}Out, layer{i+1}In, "
            
        arg_list += "fwdOut"
        return f"forward({arg_list}) ;\n"
    
    def get_loss_call(self) :
        loss = "CE" if self.loss == "CE" else "MSE"
        return f"compute{loss}Loss(BATCH, {self.net.no_class}, target, fwdOut, loss) ;\n"
    
    def get_backward_call(self) :
        net_len = len(self.net)
        arg_list = "target, fwdOut"
        
        for i in range(1, net_len+1) :
            arg_list += f", layer{i}W, layer{i}b"
            
        arg_list += f", inp"
        for i in range(1, net_len) :
            arg_list += f", layer{i}ReluHot, layer{i}Out, layer{i+1}In"

        if self.momentum :
            for i in range(1, net_len+1) :
                arg_list += f", layer{i}WMom, layer{i}bMom"
        
        return f"backward({arg_list}) ;\n"
        
    def get_training_loop(self) :
        brace_open, brace_close = '{', '}'
        return f"for i=[0:iters] {brace_open}\n\
            {self.get_intermediate_decls()}\n\
            {self.get_forward_call()}\n\
            {self.get_loss_call()}\n\
            output(ALL, loss[0]) ;\n\
            {self.get_backward_call()}\n\
        {brace_close} ;"
    
    def get_main(self) :
        brace_open, brace_close = '{', '}'
        iter_decl = f"int32_pl iters = {self.iters};\n"
        return f"\
def void main () {brace_open}\n\
{self.get_inputs()}\n\
{self.get_mom_decls() if self.momentum else ''}\n\
{iter_decl}\n\
{self.get_training_loop()}\n\
{brace_close}"
        
    def get_whole_program(self) :
        decl = self.get_batch_decl()
        fwd = self.get_forward_func()
        back = self.get_backward_func()
        return f"\
{decl}\n\n\
{fwd}\n\n\
{back}\n\n\
{self.get_main()}"
        

def torch_ffnn_to_network_args(torch_net) :
    assert isinstance(torch_net, nn.Module)

    input_size = None
    hidden_sizes = []
    output_size = None
    for p in [p.shape for p in torch_net.parameters()] :
        if input_size is None :
            input_size = p[1]
        else :
            if len(p) == 1 :
                output_size = p[0]
                continue

            hidden_sizes += [p[1]]

    return input_size, hidden_sizes, output_size


def get_translator(torch_net, batch, iters, lr, loss, momentum=False, name="") :
    net_args = torch_ffnn_to_network_args(torch_net)
    net = Network(*net_args)
    return BeaconTranslator(net=net, batch=batch, iters=iters, lr=lr, loss=loss, momentum=momentum, name=name)


def dump_ezpc(trans) :
    with open("funcs.ezpc", 'r') as f :
        s = f.read()
        f.close()
        
    with open(f"{trans.name}.ezpc", 'w') as f :
        f.write(s + '\n' + trans.get_whole_program())
        f.close()
        
    print("EzPC file dumped")
