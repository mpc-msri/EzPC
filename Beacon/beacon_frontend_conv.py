from torch import nn as nn
import sys

class ConvLayer :
    def __init__(self, kernel, in_channel, out_channel, stride, pad, pool_kernel, pool_stride, layer_no) :
        assert 0 < layer_no
        assert 0 < in_channel
        assert 0 < out_channel

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.layer_no = layer_no

    def __str__(self) :
        return f"ConvLayer({self.in_channel}, {self.out_channel}, {self.kernel}, {self.stride}, {self.pad})"
        
    def get_dims(self) :
        return (self.kernel, self.kernel, self.in_channel, self.out_channel)
    
    def get_wt_type(self, transpose=False) :
        k, k, inc, outc = self.get_dims()
        return f"float_fl[{k}][{k}][{inc}][{outc}]"
    
    def get_bias_type(self) :
        _, _, _, outc = self.get_dims()
        return f"float_fl[{outc}]"
    
    def get_wt_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}W, {self.get_wt_type()}) ; \n"
    
    def get_bias_input_stmt(self) :
        return f"input(SERVER, layer{self.layer_no}b, {self.get_bias_type()}) ; \n"

class FCLayer :
    def __init__(self, in_features, out_features, layer_no) :
        assert 0 < layer_no
        assert 0 < in_features
        assert 0 < out_features

        self.in_features = in_features
        self.out_features = out_features
        self.layer_no = layer_no

    def __str__(self) :
        return f"Layer({self.in_features}, {self.out_features})"
        
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

class ConvNetwork :
    def __init__(self, in_img, conv_layers, fc_layers, no_out, pool=2) :
        self.in_chan = conv_layers[0][1]
        self.in_img = in_img
        self.no_out = no_out
        self.pool = pool

        i = 0
        self.conv_layers = []
        for arg in conv_layers :
            i += 1
            arg += (i, )
            self.conv_layers += [ConvLayer(*arg)]

        self.fc_layers = []
        for arg in fc_layers :
            i += 1
            arg += (i, )
            self.fc_layers += [FCLayer(*arg)]

    @property
    def no_class(self) :
        return self.no_out

    @property
    def conv_len(self) :
        return len(self.conv_layers)

    @property
    def fc_len(self) :
        return len(self.fc_layers)
        
    def __len__(self) :
        return len(self.conv_layers) + len(self.fc_layers)
        
    def __getitem__(self, n) :
        return self.conv_layers[n] if n < self.conv_len else self.fc_layers[n-self.conv_len]
    
    def __str__(self) :
        str1 = "Neural network of the following layers - \n"
        for l in self.conv_layers :
            str1 += f"\t{l}\n"
        for l in self.fc_layers :
            str1 += f"\t{l}\n"
        return str1

class BeaconTranslator :
    def __get_img_sizes(self) :
        self.img_sizes = []
        img = self.in_img
        for ind, l in enumerate(self.net.conv_layers) :
            k, k, in_c, out_c = l.get_dims()
            img1 = (img + 2*l.pad - k)//l.stride + 1
            img2 = (img1 - l.pool_kernel)//l.pool_stride + 1
            img_poolmask = img2*l.pool_kernel

            self.img_sizes.append((img, img1, img2, img_poolmask))
            img = img2

    def __init__(self, tnet, net, in_img, in_chan, batch, iters, lr, loss="CE", momentum=False, name="") :
        self.batch = batch
        self.in_img = in_img
        self.in_chan = in_chan
        self.iters = iters
        self.tnet = tnet
        self.net = net
        self.lr = lr
        self.loss = loss
        self.momentum = momentum
        self.name = name

        self.__get_img_sizes()
        
    def __str__(self) :
        str1 = f"Input image {self.in_img} using a batch size of {self.batch} to train for {self.iters} iterations with lr={self.lr}\n"
        str1 += str(net)
        return str1
    

    def get_batch_decl(self) :
        str1 = f"int32 BATCH={self.batch} ;\n"
        return str1


    def get_in_img_type(self) :
        return f"float_fl[BATCH][{self.in_img}][{self.in_img}][{self.net.in_chan}]"

    
    def get_label_type(self) :
        return f"float_fl[BATCH][{self.net.no_class}]"


    def get_in_img_input_stmt(self) :
        return f"input(CLIENT, inp, {self.get_in_img_type()}) ;\n"


    def get_label_input_stmt(self) :
        return f"input(CLIENT, target, {self.get_label_type()}) ;\n"

    def get_mom_decls(self) :
        mom_str = ""
        for l in self.net.conv_layers + self.net.fc_layers :
            ind = l.layer_no
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            mom_str += f"\
{wt_type} layer{ind}WMom ;\n\
{bias_type} layer{ind}bMom ;\n\
"
        return mom_str

    def get_intermediate_decls(self) :
        decls = ""
        decls += f"\
    float_fl[BATCH][{self.in_img}][{self.in_img}][{self.net.in_chan}] layer1In ;\n\
"
        # Convolution Layers
        img = self.in_img
        for ind, l in enumerate(self.net.conv_layers) :
            k, k, in_c, out_c = l.get_dims()
            img1 = (img + 2*l.pad - k)//l.stride + 1
            img2 = (img1 - l.pool_kernel)//l.pool_stride + 1
            img_poolmask = img2*l.pool_kernel

            decls += f"\
    bool_bl[BATCH][{img1}][{img1}][{out_c}] layer{l.layer_no}Hot ;\n\
    float_fl[BATCH][{img1}][{img1}][{out_c}] layer{l.layer_no}Out ;\n\
    bool_bl[BATCH][{img_poolmask}][{img_poolmask}][{out_c}] layer{l.layer_no}Pool ;\n\
"
            if ind < self.net.conv_len - 1 :
                decls += f"\
    float_fl[BATCH][{img2}][{img2}][{out_c}] layer{l.layer_no + 1}In ;\n\
"
            img = img2

        # FC Layer intermediates
        for l in self.net.fc_layers[:-1] :
            decls += f"\
    float_fl[BATCH][{l.in_features}] layer{l.layer_no}In ;\n\
    bool_bl[BATCH][{l.out_features}] layer{l.layer_no}Hot ;\n\
    float_fl[BATCH][{l.out_features}] layer{l.layer_no}Out ;\n\
"

        # Last layer
        last_layer = self.net.fc_layers[-1]
        decls += f"\
    float_fl[BATCH][{last_layer.in_features}] layer{last_layer.layer_no}In ;\n\
"

        decls += f"\
    float_fl[BATCH][{self.net.no_class}] fwdOut ;\n\
    float_fl[1] loss ;\n\
"

        return decls

    def get_header_lst(self) :
        """ Common function signature arguments to both forward and backward calls """

        header_list = ""

        # Conv Layers
        for l in self.net.conv_layers :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            if header_list == "" :
                header_list += f"{wt_type} layer{l.layer_no}W, {bias_type} layer{l.layer_no}b"
            else :
                header_list += f", {wt_type} layer{l.layer_no}W, {bias_type} layer{l.layer_no}b"

        # FC Layers
        for l in self.net.fc_layers :
            wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
            header_list += f", {wt_type} layer{l.layer_no}W, {bias_type} layer{l.layer_no}b"

        # Network input image
        header_list += f", float_fl[BATCH][{self.in_img}][{self.in_img}][{self.net.in_chan}] layer1In"

        # Conv Layer intermediates
        img = self.in_img
        for ind, l in enumerate(self.net.conv_layers) :
            k, k, in_c, out_c = l.get_dims()
            img1 = (img + 2*l.pad - k)//l.stride + 1
            img2 = (img1 - l.pool_kernel)//l.pool_stride + 1
            img_poolmask = img2*l.pool_kernel

            header_list += f"\
, bool_bl[BATCH][{img1}][{img1}][{out_c}] layer{l.layer_no}Hot\
, float_fl[BATCH][{img1}][{img1}][{out_c}] layer{l.layer_no}Out\
, bool_bl[BATCH][{img_poolmask}][{img_poolmask}][{out_c}] layer{l.layer_no}Pool\
"
            if ind < self.net.conv_len - 1 :
                header_list += f"\
, float_fl[BATCH][{img2}][{img2}][{out_c}] layer{l.layer_no + 1}In\
"
            img = img2

        # FC Layer intermediates
        for l in self.net.fc_layers[:-1] :
            header_list += f"\
, float_fl[BATCH][{l.in_features}] layer{l.layer_no}In\
, bool_bl[BATCH][{l.out_features}] layer{l.layer_no}Hot\
, float_fl[BATCH][{l.out_features}] layer{l.layer_no}Out\
"

        # Last layer
        last_layer = self.net.fc_layers[-1]
        header_list += f", float_fl[BATCH][{last_layer.in_features}] layer{last_layer.layer_no}In"

        header_list += f", float_fl[BATCH][{self.net.no_class}] fwdOut"
        return header_list

    def get_forward_header(self) : 
        header_list = self.get_header_lst()
        return f"def void forward({header_list})"


    def get_backward_header(self) :
        header_list = self.get_header_lst()

        if self.momentum :
            for l in self.net.conv_layers + self.net.fc_layers :
                ind = l.layer_no
                wt_type, bias_type = l.get_wt_type(), l.get_bias_type()
                header_list += f", {wt_type} layer{ind}WMom, {bias_type} layer{ind}bMom"

        header_list += f", {self.get_label_type()} target"
        return f"def void backward({header_list})"


    def get_forward_body(self) :
        body = ""
        img = self.in_img
        for ind, l in enumerate(self.net.conv_layers) :
            k, k, in_c, out_c = l.get_dims()
            lno = l.layer_no
            img1 = (img + 2*l.pad - k)//l.stride + 1
            img2 = (img1 - l.pool_kernel)//l.pool_stride + 1
            layer_str = f"layer{lno}"
            body += f"\n\
float_fl[BATCH][{img1}][{img1}][{out_c}] {layer_str}Tmp ;\n\
Conv2DGroupWrapper(BATCH, {img}, {img}, {in_c}, {k}, {k}, {out_c}, {l.pad}, {l.pad}, {l.pad}, {l.pad}, {l.stride}, {l.stride}, 1, {layer_str}In, {layer_str}W, {layer_str}Tmp) ;\n\
ConvAdd(BATCH, {img1}, {img1}, {out_c}, {layer_str}Tmp, {layer_str}b, {layer_str}Tmp) ;\n\
Relu4(BATCH, {img1}, {img1}, {out_c}, {layer_str}Tmp, {layer_str}Out, {layer_str}Hot) ;\
"

            if ind < self.net.conv_len - 1 :
                last_in = f"layer{l.layer_no+1}In"
            else :
                last_in = f"layerLastPool"
                body += f"\n\
float_fl[BATCH][{img2}][{img2}][{out_c}] layerLastPool ;\
"

            body += f"\n\
MaxPool(BATCH, {img1}, {img1}, {out_c}, {l.pool_kernel}, {l.pool_kernel}, {l.pool_stride}, {l.pool_stride}, {img2}, {img2}, {layer_str}Out, {layer_str}Pool, {last_in}) ;\n\
"

            img = img2

        # print(f"Before flattening, the img size --> {img}")
        self.final_img = img

        first_fc = self.net.fc_layers[0]
        in_f, out_f = first_fc.get_dims()

        body += f"\n\
Flatten(BATCH, {img}, {img}, {out_c}, BATCH, {in_f}, layerLastPool, layer{lno+1}In) ;\n\
"

        for l in self.net.fc_layers[:-1] :
            inf, outf = l.get_dims()
            body += f"\
{l.get_wt_type(transpose=True)} layer{l.layer_no}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{l.layer_no}Temp ;\n\
Transpose({inf}, {outf}, layer{l.layer_no}W, layer{l.layer_no}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{l.layer_no}In, layer{l.layer_no}WReshaped, layer{l.layer_no}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{l.layer_no}Temp, layer{l.layer_no}b, layer{l.layer_no}Out) ;\n\
Relu2(BATCH, {outf}, layer{l.layer_no}Out, layer{l.layer_no+1}In, layer{l.layer_no}Hot) ;\n\
\n"

        l = self.net.fc_layers[-1]
        inf, outf = l.get_dims()
        output_line_args = f"BATCH, {outf}, layer{l.layer_no}Temp, fwdOut"
        output_line = f"Softmax2({output_line_args}) ;\n" if self.loss == "CE" else f"Reassign2({self.batch}, {self.net.no_out}, layer{l.layer_no}Temp, fwdOut) ;\n"
        body += f"\
{l.get_wt_type(transpose=True)} layer{l.layer_no}WReshaped ;\n\
float_fl[BATCH][{outf}] layer{l.layer_no}Temp ;\n\
Transpose({inf}, {outf}, layer{l.layer_no}W, layer{l.layer_no}WReshaped) ;\n\
MatMul(BATCH, {inf}, {outf}, layer{l.layer_no}In, layer{l.layer_no}WReshaped, layer{l.layer_no}Temp) ;\n\
GemmAdd(BATCH, {outf}, layer{l.layer_no}Temp, layer{l.layer_no}b, layer{l.layer_no}Temp) ;\n\
{output_line}"
        
        return body


    def get_forward_func(self) :
        brace_open = '{'
        brace_close = '}'
        return f"{self.get_forward_header()} {brace_open}\n\
{self.get_forward_body()}\n\
{brace_close}"


    def get_inputs(self) :
        # str_inp = f"input(CLIENT, inp, float_fl[BATCH][{self.in_img}][{self.in_img}][{self.net.in_chan}]) ;\n"
        # str_lab = f"input(CLIENT, target, float_fl[BATCH][{self.net.no_class}]) ;\n"
        str_inp = self.get_in_img_input_stmt()
        str_lab = self.get_label_input_stmt()
        str_ret = str_inp + str_lab

        for l in self.net.conv_layers :
            str_ret += l.get_wt_input_stmt()
            str_ret += l.get_bias_input_stmt()
        
        for l in self.net.fc_layers :
            str_ret += l.get_wt_input_stmt()
            str_ret += l.get_bias_input_stmt()
            
        return str_ret
    

    def get_loss_call(self) :
        loss = "CE" if self.loss == "CE" else "MSE"
        return f"compute{loss}Loss(BATCH, {self.net.no_class}, target, fwdOut, loss) ;\n"


    def get_call_args(self) :
        """ Common arguments for both forward and backward call """

        arg_lst = ""

        # Conv Layers
        for l in self.net.conv_layers :
            if arg_lst == "" :
                arg_lst += f"layer{l.layer_no}W, layer{l.layer_no}b"
            else :
                arg_lst += f", layer{l.layer_no}W, layer{l.layer_no}b"

        # FC Layers
        for l in self.net.fc_layers :
            arg_lst += f", layer{l.layer_no}W, layer{l.layer_no}b"

        # Network input image
        arg_lst += f", layer1In"

        # Conv Layer intermediates
        for ind, l in enumerate(self.net.conv_layers) :
            arg_lst += f", layer{l.layer_no}Hot, layer{l.layer_no}Out, layer{l.layer_no}Pool"
            if ind < self.net.conv_len - 1 :
                arg_lst += f", layer{l.layer_no + 1}In"

        # FC Layer intermediates
        for l in self.net.fc_layers[:-1] :
            arg_lst += f", layer{l.layer_no}In, layer{l.layer_no}Hot, layer{l.layer_no}Out"

        # Last layer
        last_layer = self.net.fc_layers[-1]
        arg_lst += f", layer{last_layer.layer_no}In"

        # Output of forward pass
        arg_lst += f", fwdOut"
        return arg_lst


    def get_forward_call(self) :
        arg_lst = self.get_call_args()
        return f"forward({arg_lst}) ;\n"


    def get_backward_call(self) :
        arg_lst = self.get_call_args()

        if self.momentum :
            for l in self.net.conv_layers + self.net.fc_layers :
                ind = l.layer_no
                arg_lst += f", layer{ind}WMom, layer{ind}bMom"

        # Target and network output
        arg_lst += f", target"
        return f"backward({arg_lst}) ;\n"

    def get_backward_body(self) :
        backward_body = ""

        # Backward pass through fully connected layers
        for l in reversed(self.net.fc_layers) :
            inf, outf = l.get_dims()
            ind = l.layer_no

            actDer_decl_type = f"float_fl[BATCH][{inf}]"
            if ind > self.net.conv_len + 1 :
                actDer_decl_name = f"layer{ind-1}ActDer"
            else :
                actDer_decl_name = "flatActDer"
            actDer_decl = f"{actDer_decl_type} {actDer_decl_name} ;\n"

            if ind == self.net.conv_len + 1 :
                last_conv = self.net.conv_layers[-1]
                _, _, _, out_c = last_conv.get_dims()
                actDer_decl += f"float_fl[BATCH][{self.final_img}][{self.final_img}][{out_c}] layer{ind-1}PooledDer ;\n"
            
            layer_decls = f"\
float_fl[BATCH][{outf}] layer{ind}Der ;\n\
float_fl[{inf}][BATCH] layer{ind}InReshaped ;\n\
float_fl[{inf}][{outf}] layer{ind}WDerReshaped ;\n\
float_fl[{outf}] layer{ind}bDer ;\n\
{actDer_decl}"
            
            arg1 = "BATCH"
            arg2 = outf
            arg3 = "fwdOut" if ind == len(self.net) else f"layer{ind}ActDer"
            arg4 = "target" if ind == len(self.net) else f"layer{ind}Hot"
            arg5 = f"layer{ind}Der"
            arg_list = f"{arg1}, {arg2}, {arg3}, {arg4}, {arg5}"

            arg_list = arg_list + (", true" if ind != len(self.net) else '')
            func_name = "getOutDer" if ind == len(self.net) else "IfElse2"
            
            act_call = f"{func_name}({arg_list}) ;\n"
            if ind > self.net.conv_len + 1 :
                firstFCActDerName = f"layer{ind-1}ActDer"
            else :
                firstFCActDerName = f"flatActDer"
                
            actDer_call = f"\
MatMul(BATCH, {outf}, {inf}, layer{ind}Der, layer{ind}W, {firstFCActDerName}) ;\n\
"

            layer_calls = f"\
{act_call}\
Transpose({inf}, BATCH, layer{ind}In, layer{ind}InReshaped) ;\n\
MatMul({inf}, BATCH, {outf}, layer{ind}InReshaped, layer{ind}Der, layer{ind}WDerReshaped) ;\n\
getBiasDer(BATCH, {outf}, layer{ind}Der, layer{ind}bDer) ;\n\
{actDer_call}"
            
            layer_body = f"{layer_decls}{layer_calls}\n"
            backward_body += layer_body

        flatten_dim = inf
        img = self.final_img
        # Backward pass through convolution layers
        for l, img_pkt in zip(reversed(self.net.conv_layers), reversed(self.img_sizes)) :
            ind = l.layer_no
            k, k, in_c, out_c = l.get_dims()
            img, img1, img2, img_poolmask = img_pkt
            img2, img1, img = img, img1, img2       # swap (img, img2)

            # img1 = (img-1)*l.pool_stride + l.pool_kernel
            # img2 = (img1-1)*l.stride + k - 2*l.pad

            pooled_der_decl = f"float_fl[BATCH][{img2}][{img2}][{in_c}] layer{ind-1}PooledDer ;\n" if ind > 1 else ''
            layer_decls = f"\
float_fl[BATCH][{img1}][{img1}][{out_c}] layer{ind}ExpandedPooledDer ;\n\
float_fl[BATCH][{img1}][{img1}][{out_c}] layer{ind}ActDer ;\n\
float_fl[BATCH][{img1}][{img1}][{out_c}] layer{ind}Der ;\n\
float_fl[{k}][{k}][{in_c}][{out_c}] layer{ind}WDer ;\n\
float_fl[{out_c}] layer{ind}bDer ;\n\
{pooled_der_decl}\
"
            backward_body += layer_decls

            layer_calls = ""
            if ind == self.net.conv_len :
                layer_calls += f"Unflatten(BATCH, {flatten_dim}, BATCH, {self.final_img}, {self.final_img}, {out_c}, flatActDer, layer{ind}PooledDer) ;\n"

            pooled_der_call = f"GetPooledDer(BATCH, {img2}, {img2}, {in_c}, {out_c}, {img1}, {img1}, {k}, {k}, layer{ind}W, layer{ind}Der, layer{ind-1}PooledDer) ;\n" if ind > 1 else ''
            if l.pool_kernel == l.pool_stride :
                pool_line = f"\
PoolExpand(BATCH, {img}, {img}, {out_c}, {l.pool_kernel}, {l.pool_kernel}, {img1}, {img1}, layer{ind}PooledDer, layer{ind}ExpandedPooledDer) ;\n\
IfElse4(BATCH, {img1}, {img1}, {out_c}, layer{ind}ExpandedPooledDer, layer{ind}Pool, layer{ind}ActDer, false) ;\
"
            else :
                pool_line = f"PoolProp(BATCH, {out_c}, {img}, {img_poolmask}, {img1}, {l.pool_kernel}, {l.pool_stride}, layer{ind}PooledDer, layer{ind}Pool, layer{ind}ActDer, false) ;"
            layer_calls += f"\
{pool_line}\n\
IfElse4(BATCH, {img1}, {img1}, {out_c}, layer{ind}ActDer, layer{ind}Hot, layer{ind}Der, true) ; \n\
ConvDerWrapper(BATCH, {img2}, {img2}, {in_c}, {k}, {k}, {out_c}, {l.pad}, {l.pad}, {l.pad}, {l.pad}, {l.stride}, {l.stride}, 1, layer{ind}In, layer{ind}WDer, layer{ind}Der) ;\n\
ConvBiasDer(BATCH, {img1}, {img1}, {out_c}, layer{ind}Der, layer{ind}bDer) ;\n\
{pooled_der_call}\n\
"
        
            img = img2
            backward_body += layer_calls

        transpose_decls = ""
        transpose_calls = ""
        update_calls = ""
        mom_factor = 'Momentum' if self.momentum else ''
        for l in self.net.conv_layers :
            ind = l.layer_no
            k, k, in_c, out_c = l.get_dims()
            update_calls += f"\
updateWeights{mom_factor}4({k}, {k}, {in_c}, {out_c}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind}W, layer{ind}WDer{', layer' + str(ind) + 'WMom' if self.momentum else ''}) ;\n\
updateWeights{mom_factor}({out_c}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind}b, layer{ind}bDer{', layer' + str(ind) + 'bMom' if self.momentum else ''}) ;\n\
"

        for l in self.net.fc_layers :
            ind = l.layer_no
            inf, outf = l.get_dims()
            transpose_decls += f"{l.get_wt_type()} layer{ind}WDer ;\n"
            transpose_calls += f"Transpose({outf}, {inf}, layer{ind}WDerReshaped, layer{ind}WDer) ;\n"
            update_calls += f"\
updateWeights{mom_factor}2({outf}, {inf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind}W, layer{ind}WDer{', layer' + str(ind) + 'WMom' if self.momentum else ''}) ;\n\
updateWeights{mom_factor}({outf}, {self.lr}, {'0.9, ' if self.momentum else ''}layer{ind}b, layer{ind}bDer{', layer' + str(ind) + 'bMom' if self.momentum else ''}) ;\n\
"

        update_body = f"{transpose_decls}{transpose_calls}\n{update_calls}"
        backward_body += update_body
        return backward_body


    def get_backward_func(self) :
        brace_open = '{'
        brace_close = '}'
        return f"{self.get_backward_header()} {brace_open}\n\
{self.get_backward_body()}\n\
{brace_close}"


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

    conv_sizes = []
    fc_sizes = []
    output_size = None

    for pr in torch_net.params :
        if (type(pr) == type(())) :
            pool_kernel, pool_stride = pr[1], pr[2]
            pr = pr[0]
            
        wt = pr.weight
        bs = pr.bias
        sw = wt.shape
        sb = bs.shape

        if len(sw) == 4 :
            if pr.padding != (0, 0) :
                p, _ = pr.padding
                s, _ = pr.stride
                conv_sizes += [(sw[2], sw[1], sw[0], s, p, pool_kernel, pool_stride)]
            else :
                conv_sizes += [(sw[2], sw[1], sw[0], 1, 0, pool_kernel, pool_stride)]
        elif len(sw) == 2 :
            fc_sizes += [(sw[1], sw[0])]

        output_size = sb[0]

    return conv_sizes, fc_sizes, output_size

def get_translator(torch_net, in_img, in_chan, batch, iters, lr, loss, momentum=False, name="") :
    net_args = torch_ffnn_to_network_args(torch_net)
    net_args = (in_img, ) + net_args
    net = ConvNetwork(*net_args)
    return BeaconTranslator(tnet=torch_net, net=net, in_img=in_img, in_chan=in_chan, batch=batch, iters=iters, lr=lr, loss=loss, momentum=momentum, name=name)

def dump_ezpc(trans) :
    with open("funcs.ezpc", 'r') as f :
        s = f.read()
        f.close()
        
    with open(f"{trans.name}.ezpc", 'w') as f :
        f.write(s + '\n' + trans.get_whole_program())
        f.close()
        
    print("EzPC file dumped")
