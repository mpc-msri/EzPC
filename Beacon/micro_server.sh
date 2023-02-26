printf "RUNNING : Accumulate\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/accum_secfloat r=1 nt=16 sz1=2000 sz2=2000
printf "Beacon FP32\n"
./../SCI/build/bin/accum_beacon r=1 nt=16 sz1=2000 sz2=2000 chunk=26
# ./../SCI/build/bin/accum_beacon r=1 nt=16 sz1=2000 sz2=2000 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/accum_beacon r=1 nt=16 sz1=2000 sz2=2000 chunk=26 mbits=7 ebits=8
printf "||-------------------||\n"

printf "RUNNING : DotProd\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/dotprod_secfloat r=1 nt=16 sz1=1000 sz2=1000
printf "Beacon FP32\n"
./../SCI/build/bin/dotprod_beacon r=1 nt=16 sz1=1000 sz2=1000 chunk=26
# ./../SCI/build/bin/dotprod_beacon r=1 nt=16 sz1=1000 sz2=1000 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/dotprod_beacon r=1 nt=16 sz1=1000 sz2=1000 chunk=26 mbits=7 ebits=8
printf "||-------------------||\n"

printf "RUNNING : MatMul\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/matmul_secfloat r=1 nt=16 sz1=100 sz2=100 sz3=100
printf "Beacon FP32\n"
./../SCI/build/bin/matmul_beacon r=1 nt=16 sz1=100 sz2=100 sz3=100 chunk=26
# ./../SCI/build/bin/matmul_beacon r=1 nt=16 sz1=100 sz2=100 sz3=100 chunk=26 mbits=10 ebits=8
# ./../SCI/build/bin/matmul_beacon r=1 nt=16 sz1=100 sz2=100 sz3=100 chunk=26 mbits=7 ebits=8
printf "||-------------------||\n"

printf "RUNNING : Softmax\n"
printf "Secfloat FP32\n"
./../SCI/build/bin/softmax_secfloat r=1 nt=16 sz1=1000 sz2=100
printf "Beacon FP32\n"
./../SCI/build/bin/softmax_beacon r=1 nt=16 sz1=1000 sz2=100 chunk=26
printf "||-------------------||\n"

printf "RUNNING : Sigmoid\n"
printf "Sigmoid FP32\n"
./../SCI/build/bin/sigmoid_secfloat r=1 nt=16 sz1=1000000
printf "Beacon FP32\n"
./../SCI/build/bin/sigmoid_beacon r=1 nt=16 sz1=1000000 chunk=26
printf "||-------------------||\n"
