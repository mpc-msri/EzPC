import torch
import argparse
import os

nn = torch.nn
F = nn.functional
optim = torch.optim
torch.set_printoptions(precision=30)
dtype = torch.float32

import beacon_frontend as bcf
import beacon_frontend_conv as bcf_conv

## Model definitions


class LeNet(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=1, dtype=dtype
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, dtype=dtype
        )

        self.fc1 = nn.Linear(400, 120, dtype=dtype)
        self.fc2 = nn.Linear(120, 84, dtype=dtype)
        self.fc3 = nn.Linear(84, 10, dtype=dtype)

        self.params = [
            (self.conv1, 2, 2),
            (self.conv2, 2, 2),
            self.fc1,
            self.fc2,
            self.fc3,
        ]

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class HiNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(HiNet, self).__init__()

        self.pool = nn.MaxPool2d(3, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 5, 1, 1)
        self.fc1 = nn.Linear(64, 10)

        self.params = [
            (self.conv1, 3, 2),
            (self.conv2, 3, 2),
            (self.conv3, 3, 2),
            self.fc1,
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 64)
        x = self.fc1(x)
        return x


class Relevance(nn.Module):
    def __init__(self):
        super(Relevance, self).__init__()
        self.fc1 = nn.Linear(874, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class MNISTLogistic(nn.Module):
    def __init__(self):
        super(MNISTLogistic, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)


class MNISTFFNN(nn.Module):
    def __init__(self):
        super(MNISTFFNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


## Dumping weights and input


class Ez_Network(nn.Module):
    def __init__(self, layers, log=False):
        super(Ez_Network, self).__init__()
        self.log = log
        self.layers = nn.ModuleList(
            [nn.Linear(*l.get_dims()).to(dtype) for l in layers]
        )

    def forward(self, x):
        lst = []
        for h in self.layers[:-1]:
            x = torch.mm(x, h.weight.T)
            lst += [x]
            x = x + h.bias
            lst += [x]
            x = F.relu(x)
            lst += [x]

        l = self.layers[-1]
        x = torch.mm(x, l.weight.T)
        lst += [x]
        x = x + l.bias
        lst += [x]
        return (x, lst) if self.log else x


def get_pytorch_stuff_ffnn(ez, log=False):
    torch.manual_seed(0)
    net = Ez_Network(ez.net.layers, log)
    with open(f"{ez.name}_weights.inp", "w") as f:
        for p in net.parameters():
            for el in p.flatten():
                f.write(str(el.item()) + "\n")
        f.close()

    torch.manual_seed(44)
    inp = torch.rand(ez.batch, ez.net.in_dim).to(dtype)
    net_inp = 2 * inp - 1
    with open(f"{ez.name}_input{ez.batch}.inp", "w") as f:
        for el in net_inp.flatten():
            f.write(str(el.item()) + "\n")
        f.close()

    torch.manual_seed(44)
    if ez.loss == "CE":
        randperm = torch.cat(
            [torch.randperm(ez.net.no_class) for _ in range(ez.batch)]
        ).reshape(ez.batch, ez.net.no_class)

        lab_out = torch.cat(
            [(row == ez.net.no_class - 1).to(torch.int64) for row in randperm]
        ).reshape(ez.batch, ez.net.no_class)

        with open(f"{ez.name}_labels{ez.batch}.inp", "w") as f:
            for el in lab_out.flatten():
                f.write(str(el.item()) + "\n")
            f.close()

        net_out = torch.stack([torch.argmax(row) for row in randperm])
    else:
        target_vals = 10 * torch.rand(ez.batch, 1).to(dtype)
        with open(f"{ez.name}_target{ez.batch}.inp", "w") as f:
            for el in target_vals.flatten():
                f.write(str(el.item()) + "\n")
            f.close()

        net_out = target_vals

    return net, net_inp, net_out


def get_pytorch_stuff_conv(ez, log=False):
    torch.manual_seed(0)
    net = ez.net
    tnet = ez.tnet
    with open(f"{ez.name}_weights.inp", "w") as f:
        for p in tnet.parameters():
            for el in p.flatten():
                f.write(str(el.item()) + "\n")
        f.close()

    torch.manual_seed(44)
    inp = torch.rand(ez.batch, ez.in_chan, ez.in_img, ez.in_img).to(dtype)
    net_inp = 2 * inp - 1
    with open(f"{ez.name}_input{ez.batch}.inp", "w") as f:
        for el in net_inp.flatten():
            f.write(str(el.item()) + "\n")
        f.close()

    torch.manual_seed(44)
    if ez.loss == "CE":
        randperm = torch.cat(
            [torch.randperm(ez.net.no_class) for _ in range(ez.batch)]
        ).reshape(ez.batch, ez.net.no_class)

        lab_out = torch.cat(
            [(row == ez.net.no_class - 1).to(torch.int64) for row in randperm]
        ).reshape(ez.batch, ez.net.no_class)

        with open(f"{ez.name}_labels{ez.batch}.inp", "w") as f:
            for el in lab_out.flatten():
                f.write(str(el.item()) + "\n")
            f.close()

        net_out = torch.stack([torch.argmax(row) for row in randperm])
    else:
        target_vals = 10 * torch.rand(ez.batch, 1).to(dtype)
        with open(f"{ez.name}_target{ez.batch}.inp", "w") as f:
            for el in target_vals.flatten():
                f.write(str(el.item()) + "\n")
            f.close()

        net_out = target_vals

    return net, net_inp, net_out


def do_the_cmake(name):
    bo, bc = "{", "}"
    dollah = "$"
    text = f'\
cmake_minimum_required (VERSION 3.0)\n\
project (MY_PROJ)\n\
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/Beacon/SCI/build/install")\n\
\n\
macro(add_network_secfloat name)\n\
	add_executable(${bo}name{bc}_secfloat "{dollah}{bo}name{bc}.cpp")\n\
	target_link_libraries({dollah}{bo}name{bc}_secfloat SCI::SCI-FloatML)\n\
	target_compile_options({dollah}{bo}name{bc}_secfloat PRIVATE "-w")\n\
endmacro()\n\
\n\
macro(add_network_beacon name)\n\
	add_executable({dollah}{bo}name{bc}_beacon "{dollah}{bo}name{bc}.cpp")\n\
	target_link_libraries({dollah}{bo}name{bc}_beacon SCI::SCI-FloatBeacon)\n\
	target_compile_options({dollah}{bo}name{bc}_beacon PRIVATE "-w")\n\
endmacro()\n\
\n\
add_network_secfloat({name})\n\
add_network_beacon({name})\n\
'

    with open("CMakeLists.txt", "w") as file:
        file.write(text)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BeaconFrontend",
        description="Compile networks for Beacon",
        epilog="Call as \n>python3 compile_networks.py <name of network>",
    )

    parser.add_argument("network")
    parser.add_argument("batch")
    parser.add_argument("iters")
    parser.add_argument("lr")
    parser.add_argument("loss")
    parser.add_argument("momentum")
    args = parser.parse_args()
    print(args.network)

    if args.network in ["Relevance", "Logistic", "FFNN"]:
        net = {"Relevance": Relevance, "Logistic": MNISTLogistic, "FFNN": MNISTFFNN}[
            args.network
        ]()
        trans = bcf.get_translator(
            net,
            batch=int(args.batch),
            iters=int(args.iters),
            lr=float(args.lr),
            loss=args.loss,
            momentum=(args.momentum == "yes"),
            name=args.network,
        )
        bcf.dump_ezpc(trans)

        net1, net_inp, net_out = get_pytorch_stuff_ffnn(trans, False)
    elif args.network in ["LeNet", "HiNet"]:
        net = {"LeNet": LeNet, "HiNet": HiNet}[args.network]()
        trans = bcf_conv.get_translator(
            net,
            in_img=32,
            in_chan=3,
            batch=int(args.batch),
            iters=int(args.iters),
            lr=float(args.lr),
            loss=args.loss,
            momentum=(args.momentum == "yes"),
            name=args.network,
        )
        bcf_conv.dump_ezpc(trans)

        net1, net_inp, net_out = get_pytorch_stuff_conv(trans)
    else:
        print("Invalid network")
        sys.exit(1)

    do_the_cmake(args.network)

    os.system(f"../EzPC/EzPC/ezpc --codegen SECFLOAT --bitlen 32 {args.network}.ezpc")
    os.system(f"mv {args.network}0.cpp {args.network}.cpp")
    os.system("mkdir -p build")
    os.system("cd build && cmake .. && make -j")
    os.system("cp build/*_secfloat build/*_beacon .")
    os.system("rm -Rf build/")
    os.system(f"rm {args.network}.ezpc {args.network}.cpp")
    # os.system("cd ..")
