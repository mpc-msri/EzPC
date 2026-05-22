printf "Relevance\n"
printf "Secfloat\n"
./../SCI/build/bin/relevance32_secfloat r=1
printf "Beacon FP32\n"
./../SCI/build/bin/relevance32_beacon r=1
printf "Beacon BF16\n"
./../SCI/build/bin/relevance32_beacon r=1 mbits=7 ebits=8

printf "Logistic\n"
printf "Secfloat\n"
./../SCI/build/bin/logistic128_secfloat r=1  < ./../SCI/networks/inputs/logistic_weights.inp
printf "Beacon FP32\n"
./../SCI/build/bin/logistic128_beacon r=1  < ./../SCI/networks/inputs/logistic_weights.inp
printf "Beacon BF16\n"
./../SCI/build/bin/logistic128_beacon r=1 mbits=7 ebits=8 < ./../SCI/networks/inputs/logistic_weights.inp


printf "FFNN\n"
printf "Secfloat\n"
./../SCI/build/bin/ffnn128_secfloat r=1 < ./../SCI/networks/inputs/ffnn_weights.inp
printf "Beacon FP32\n"
./../SCI/build/bin/ffnn128_beacon r=1 < ./../SCI/networks/inputs/ffnn_weights.inp
printf "Beacon BF16\n"
./../SCI/build/bin/ffnn128_beacon r=1 mbits=7 ebits=8 < ./../SCI/networks/inputs/ffnn_weights.inp


printf "LeNet\n"
printf "Secfloat\n"
./../SCI/build/bin/lenet128_secfloat r=1 < ./../SCI/networks/inputs/lenet_weights.inp
printf "Beacon FP32\n"
./../SCI/build/bin/lenet128_beacon r=1 < ./../SCI/networks/inputs/lenet_weights.inp
printf "Beacon BF16\n"
./../SCI/build/bin/lenet128_beacon r=1 mbits=7 ebits=8 < ./../SCI/networks/inputs/lenet_weights.inp

printf "HiNet\n"
printf "Secfloat\n"
./../SCI/build/bin/hinet4_secfloat r=1 < ./../SCI/networks/inputs/hinet_weights.inp
printf "Beacon FP32\n"
./../SCI/build/bin/hinet4_beacon r=1 < ./../SCI/networks/inputs/hinet_weights.inp
printf "Beacon BF16\n"
./../SCI/build/bin/hinet4_beacon r=1 mbits=7 ebits=8 < ./../SCI/networks/inputs/hinet_weights.inp
