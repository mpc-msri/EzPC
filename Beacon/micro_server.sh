printf "RUNNING : Summation\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/vsum-secfloatml r=1 nt=16 sz1=2000 sz2=2000
printf "Beacon FP32\n"
./../SCI/build/bin/vsum-beacon r=1 nt=16 sz1=2000 sz2=2000 chunk=26
printf "Beacon BF16\n"
./../SCI/build/bin/vsum-beacon r=1 nt=16 sz1=2000 sz2=2000 chunk=26 mbits=7 ebits=8

printf "RUNNING : Dotprod\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/dotprod-secfloatml r=1 nt=16 sz1=1000 sz2=1000
printf "Beacon FP32\n"
./../SCI/build/bin/dotprod-beacon r=1 nt=16 sz1=1000 sz2=1000 chunk=26
printf "Beacon BF16\n"
./../SCI/build/bin/dotprod-beacon r=1 nt=16 sz1=1000 sz2=1000 chunk=26 mbits=7 ebits=8

printf "RUNNING : Matmul\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/matmul-secfloatml r=1 nt=16 sz1=100 sz2=100 sz3=100
printf "Beacon FP32\n"
./../SCI/build/bin/matmul-beacon r=1 nt=16 sz1=100 sz2=100 sz3=100 chunk=26
printf "Beacon BF16\n"
./../SCI/build/bin/matmul-beacon r=1 nt=16 sz1=100 sz2=100 sz3=100 chunk=26 mbits=7 ebits=8

printf "RUNNING : Softmax\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/softmax-secfloatml r=1 nt=16 sz1=1000 sz2=100
printf "Beacon FP32\n"
./../SCI/build/bin/softmax-beacon r=1 nt=16 sz1=1000 sz2=100 chunk=26
printf "Beacon BF16\n"
./../SCI/build/bin/softmax-beacon r=1 nt=16 sz1=1000 sz2=100 chunk=26 mbits=7 ebits=8

printf "RUNNING : Sigmoid\n"
printf "Sigmoid FP32\n"
./../SCI/build/bin/sigmoid-secfloatml r=1 nt=16 sz1=1000000
printf "Beacon FP32\n"
./../SCI/build/bin/sigmoid-beacon r=1 nt=16 sz1=1000000 chunk=26
printf "Beacon BF16\n"
./../SCI/build/bin/sigmoid-beacon r=1 nt=16 sz1=1000000 chunk=26 mbits=7 ebits=8