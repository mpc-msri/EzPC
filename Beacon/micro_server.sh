printf "RUNNING : Summation\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/vsum-secfloatml r=1 sz1=2000 sz2=2000
printf "Beacon FP32\n"
./../SCI/build/bin/vsum-beacon r=1 sz1=2000 sz2=2000
printf "Beacon BF16\n"
./../SCI/build/bin/vsum-beacon r=1 sz1=2000 sz2=2000 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/vsum-beacon r=1 sz1=2000 sz2=2000 mbits=10 ebits=8


printf "RUNNING : Dotprod\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/dotprod-secfloatml r=1 sz1=1000 sz2=1000
printf "Beacon FP32\n"
./../SCI/build/bin/dotprod-beacon r=1 sz1=1000 sz2=1000
printf "Beacon BF16\n"
./../SCI/build/bin/dotprod-beacon r=1 sz1=1000 sz2=1000 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/dotprod-beacon r=1 sz1=1000 sz2=1000 mbits=10 ebits=8


printf "RUNNING : Matmul\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/matmul-secfloatml r=1 sz1=100 sz2=100 sz3=100
printf "Beacon FP32\n"
./../SCI/build/bin/matmul-beacon r=1 sz1=100 sz2=100 sz3=100
printf "Beacon BF16\n"
./../SCI/build/bin/matmul-beacon r=1 sz1=100 sz2=100 sz3=100 mbits=7 ebits=8
printf "Beacon FP19\n"
./../SCI/build/bin/matmul-beacon r=1 sz1=100 sz2=100 sz3=100 mbits=10 ebits=8


printf "RUNNING : Softmax\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/softmax-secfloatml r=1 sz1=1000 sz2=100
printf "Beacon FP32\n"
./../SCI/build/bin/softmax-beacon r=1 sz1=1000 sz2=100
printf "Beacon BF16\n"
./../SCI/build/bin/softmax-beacon r=1 sz1=1000 sz2=100 mbits=7 ebits=8

printf "RUNNING : Sigmoid\n"
printf "Beacon FP32\n"
./../SCI/build/bin/sigmoid-beacon r=1 sz1=1000000
printf "Beacon BF16\n"
./../SCI/build/bin/sigmoid-beacon r=1 sz1=1000000 mbits=7 ebits=8