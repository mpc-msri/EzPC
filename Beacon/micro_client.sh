printf "RUNNING : Accumulate\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/accum_secfloat r=2 nt=16 sz1=2000 sz2=2000 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/accum_beacon r=2 nt=16 sz1=2000 sz2=2000 add=$1 chunk=26
# ./../SCI/build/bin/accum_beacon r=2 nt=16 sz1=2000 sz2=2000 add=$1 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/accum_beacon r=2 nt=16 sz1=2000 sz2=2000 add=$1 chunk=26 mbits=7 ebits=8

printf "RUNNING : Dotprod\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/dotprod_secfloat r=2 nt=16 sz1=1000 sz2=1000 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/dotprod_beacon r=2 nt=16 sz1=1000 sz2=1000 add=$1 chunk=26
# ./../SCI/build/bin/dotprod_beacon r=2 nt=16 sz1=1000 sz2=1000 add=$1 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/dotprod_beacon r=2 nt=16 sz1=1000 sz2=1000 add=$1 chunk=26 mbits=7 ebits=8

printf "RUNNING : Matmul\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/matmul_secfloat r=2 nt=16 sz1=100 sz2=100 sz3=100 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/matmul_beacon r=2 nt=16 sz1=100 sz2=100 sz3=100 add=$1 chunk=26
# ./../SCI/build/bin/matmul_beacon r=2 nt=16 sz1=100 sz2=100 sz3=100 add=$1 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/matmul_beacon r=2 nt=16 sz1=100 sz2=100 sz3=100 add=$1 chunk=26 mbits=7 ebits=8

printf "RUNNING : Softmax\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/softmax_secfloat r=2 nt=16 sz1=1000 sz2=100 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/softmax_beacon r=2 nt=16 sz1=1000 sz2=100 add=$1 chunk=26

printf "RUNNING : Sigmoid\n"
printf "Sigmoid FP32\n"
./../SCI/build/bin/sigmoid_secfloat r=2 nt=16 sz1=1000000 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/sigmoid_beacon r=2 nt=16 sz1=1000000 add=$1 chunk=26