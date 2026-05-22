printf "RUNNING : Summation\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/vsum-secfloatml r=2 sz1=2000 sz2=2000 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/vsum-beacon r=2 sz1=2000 sz2=2000 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/vsum-beacon r=2 sz1=2000 sz2=2000 add=$1 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/vsum-beacon r=2 sz1=2000 sz2=2000 add=$1 mbits=10 ebits=8

printf "RUNNING : Dotprod\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/dotprod-secfloatml r=2 sz1=1000 sz2=1000 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/dotprod-beacon r=2 sz1=1000 sz2=1000 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/dotprod-beacon r=2 sz1=1000 sz2=1000 add=$1 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/dotprod-beacon r=2 sz1=1000 sz2=1000 add=$1 mbits=10 ebits=8

printf "RUNNING : Matmul\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/matmul-secfloatml r=2 sz1=100 sz2=100 sz3=100 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/matmul-beacon r=2 sz1=100 sz2=100 sz3=100 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/matmul-beacon r=2 sz1=100 sz2=100 sz3=100 add=$1 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/matmul-beacon r=2 sz1=100 sz2=100 sz3=100 add=$1 mbits=10 ebits=8

printf "RUNNING : Softmax\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/softmax-secfloatml r=2 sz1=1000 sz2=100 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/softmax-beacon r=2 sz1=1000 sz2=100 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/softmax-beacon r=2 sz1=1000 sz2=100 add=$1 mbits=7 ebits=8

printf "RUNNING : Sigmoid\n"
printf "Beacon FP32\n"
./../SCI/build/bin/sigmoid-beacon r=2 sz1=1000000 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/sigmoid-beacon r=2 sz1=1000000 add=$1 mbits=7 ebits=8