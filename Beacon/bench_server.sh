printf "Relevance\n"
printf "Secfloat\n"
./../SCI/build/bin/relevance32_split_secfloat r=1 nt=16
printf "Beacon\n"
./../SCI/build/bin/relevance32_split_beacon r=1 nt=16 chunk=26

printf "Logistic\n"
printf "Secfloat\n"
./../SCI/build/bin/logistic128_split_secfloat r=1 nt=16  < ./../SCI/networks/inputs/logistic_weights.inp
printf "Beacon\n"
./../SCI/build/bin/logistic128_split_beacon r=1 nt=16 chunk=26  < ./../SCI/networks/inputs/logistic_weights.inp


printf "FFNN\n"
printf "Secfloat\n"
./../SCI/build/bin/ffnn128_split_secfloat r=1 nt=16 < ./../SCI/networks/inputs/ffnn_weights.inp
printf "Beacon\n"
./../SCI/build/bin/ffnn128_split_beacon r=1 nt=16 chunk=26 < ./../SCI/networks/inputs/ffnn_weights.inp


printf "LeNet\n"
printf "Secfloat\n"
./../SCI/build/bin/lenet128_split_secfloat r=1 nt=16 < ./../SCI/networks/inputs/lenet_weights.inp
printf "Beacon\n"
./../SCI/build/bin/lenet128_split_beacon r=1 nt=16 chunk=20  < ./../SCI/networks/inputs/lenet_weights.inp


printf "HiNet\n"
printf "Secfloat\n"
./../SCI/build/bin/hinet4_split_secfloat r=1 nt=16 < ./../SCI/networks/inputs/hinet_weights.inp
printf "Beacon\n"
./../SCI/build/bin/hinet4_split_beacon r=1 nt=16 chunk=26 < ./../SCI/networks/inputs/hinet_weights.inp