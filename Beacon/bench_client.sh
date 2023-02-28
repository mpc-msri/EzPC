printf "Relevance\n"
printf "Secfloat\n"
./../SCI/build/bin/relevance32_secfloat r=2 add=$1
printf "Beacon FP32\n"
./../SCI/build/bin/relevance32_beacon r=2 add=$1
printf "Beacon BF16\n"
./../SCI/build/bin/relevance32_beacon r=2 add=$1 mbits=7 ebits=8

printf "Logistic\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/logistic_input128.inp ../SCI/networks/inputs/logistic_labels128.inp  | ./../SCI/build/bin/logistic128_secfloat r=2 add=$1
printf "Beacon FP32\n"
cat ../SCI/networks/inputs/logistic_input128.inp ../SCI/networks/inputs/logistic_labels128.inp  | ./../SCI/build/bin/logistic128_beacon r=2 add=$1
printf "Beacon BF16\n"
cat ../SCI/networks/inputs/logistic_input128.inp ../SCI/networks/inputs/logistic_labels128.inp  | ./../SCI/build/bin/logistic128_beacon r=2 add=$1 mbits=7 ebits=8

printf "FFNN\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/ffnn_input128.inp ../SCI/networks/inputs/ffnn_labels128.inp | ./../SCI/build/bin/ffnn128_secfloat r=2 add=$1
printf "Beacon FP32\n"
cat ../SCI/networks/inputs/ffnn_input128.inp ../SCI/networks/inputs/ffnn_labels128.inp | ./../SCI/build/bin/ffnn128_beacon r=2 add=$1
printf "Beacon BF16\n"
cat ../SCI/networks/inputs/ffnn_input128.inp ../SCI/networks/inputs/ffnn_labels128.inp | ./../SCI/build/bin/ffnn128_beacon r=2 add=$1 mbits=7 ebits=8

printf "LeNet\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/lenet_input128.inp ../SCI/networks/inputs/lenet_labels128.inp | ./../SCI/build/bin/lenet128_secfloat r=2 add=$1
printf "Beacon FP32\n"
cat ../SCI/networks/inputs/lenet_input128.inp ../SCI/networks/inputs/lenet_labels128.inp | ./../SCI/build/bin/lenet128_beacon r=2 add=$1
printf "Beacon BF16\n"
cat ../SCI/networks/inputs/lenet_input128.inp ../SCI/networks/inputs/lenet_labels128.inp | ./../SCI/build/bin/lenet128_beacon r=2 add=$1 mbits=7 ebits=8

printf "HiNet\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/hinet_input4.inp ../SCI/networks/inputs/hinet_labels4.inp | ./../SCI/build/bin/hinet4_secfloat r=2 add=$1
printf "Beacon FP32\n"
cat ../SCI/networks/inputs/hinet_input4.inp ../SCI/networks/inputs/hinet_labels4.inp | ./../SCI/build/bin/hinet4_beacon r=2 add=$1
printf "Beacon BF16\n"
cat ../SCI/networks/inputs/hinet_input4.inp ../SCI/networks/inputs/hinet_labels4.inp | ./../SCI/build/bin/hinet4_beacon r=2 add=$1 mbits=7 ebits=8 
