printf "Relevance\n"
printf "Secfloat\n"
./../SCI/build/bin/relevance32_split_secfloat r=2 nt=16 add=10.13.0.6
printf "Beacon\n"
./../SCI/build/bin/relevance32_split_beacon r=2 nt=16 chunk=26 add=$1

printf "Logistic\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/logistic_input128.inp ../SCI/networks/inputs/logistic_labels128.inp  | ./../SCI/build/bin/logistic128_secfloat r=2 nt=16 add=$1
printf "Beacon\n"
cat ../SCI/networks/inputs/logistic_input128.inp ../SCI/networks/inputs/logistic_labels128.inp  | ./../SCI/build/bin/logistic128_beacon r=2 nt=16 add=$1 chunk=26

printf "FFNN\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/ffnn_input128.inp ../SCI/networks/inputs/ffnn_labels128.inp | ./../SCI/build/bin/ffnn128_secfloat r=2 nt=16 add=$1
printf "Beacon\n"
cat ../SCI/networks/inputs/ffnn_input128.inp ../SCI/networks/inputs/ffnn_labels128.inp | ./../SCI/build/bin/ffnn128_beacon r=2 nt=16 add=$1 chunk=26

printf "LeNet\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/lenet_input128.inp ../SCI/networks/inputs/lenet_labels128.inp | ./../SCI/build/bin/lenet128_secfloat r=2 nt=16 add=$1
printf "Beacon\n"
cat ../SCI/networks/inputs/lenet_input128.inp ../SCI/networks/inputs/lenet_labels128.inp | ./../SCI/build/bin/lenet128_beacon r=2 nt=16 add=$1 chunk=20

printf "HiNet\n"
printf "Secfloat\n"
cat ../SCI/networks/inputs/hinet_input4.inp ../SCI/networks/inputs/hinet_labels4.inp | ./../SCI/build/bin/hinet4_split_secfloat r=2 nt=16 add=$1
printf "Beacon\n"
cat ../SCI/networks/inputs/hinet_input4.inp ../SCI/networks/inputs/hinet_labels4.inp | ./../SCI/build/bin/hinet4_split_beacon r=2 nt=16 add=$1 chunk=26