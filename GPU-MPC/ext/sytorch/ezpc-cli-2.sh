#!/bin/bash

# Default values
BACKEND="LLAMA"
SCALE="15"
BITLENGTH="40"

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -m|--model)
            MODEL_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        -c|--client)
            CLIENT_IP="$2"
            shift # past argument
            shift # past value
            ;;
        -s|--server)
            SERVER_IP="$2"
            shift # past argument
            shift # past value
            ;;
        -d|--dealer)
            DEALER_IP="$2"
            shift # past argument
            shift # past value
            ;;
        -b|--backend)
            BACKEND="$2"
            shift # past argument
            shift # past value
            ;;
        -scale|--scale)
            SCALE="$2"
            shift # past argument
            shift # past value
            ;;
        -bl|--bitlength)
            BITLENGTH="$2"
            shift # past argument
            shift # past value
            ;;
        -preprocess|--preprocess)
            PREPROCESS="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check that required arguments are set
if [ -z "$MODEL_PATH" ] || [ -z "$SERVER_IP" ] || [ -z "$PREPROCESS" ] || [ -z "$DEALER_IP" ] ;
then
    echo "To run the secure MPC model, please ensure the following:"
    echo "Server Files:             Client Files:       Dealer Files:"
    echo "-------------------       --------------      --------------"
    echo "path to model.onnx        path to image.jpg   dealer IP"
    echo "path to preprocess.py)"
    echo "server IP"
    echo "-------------------       --------------" | column -t -s $'\t'
    echo "Usage: $0 -m <full-path/model.onnx> -preprocess <full-path/preprocess_image_file> -s <server-ip>  -d <dealer-ip>"
    echo "Optional: [-b <backend>] [-scale <scale>] [-bl <bitlength>]"
    exit 1
fi

# Print out arguments
echo ------------------------------
echo "SERVER Details:"
echo "Model path: $MODEL_PATH"
echo "Preprocess path: $PREPROCESS"
echo "Server IP: $SERVER_IP"
echo ------------------------------
echo "CLIENT Details:"
echo "Image path: $IMAGE_PATH"
# echo "Client IP: $CLIENT_IP"
echo ------------------------------
echo "DEALER Details:"
echo "Dealer IP: $DEALER_IP"
echo ------------------------------

# Getting Model Name and Directory and Model Name without extension
File_NAME=$(basename $MODEL_PATH)
MODEL_DIR=$(dirname $MODEL_PATH)
Model_Name=${File_NAME%.*}

# Get preprocess file name
preprocess_image_file=$(basename $PREPROCESS)

# Generating Server Script
SERVER_SCRIPT="server.sh"
echo "Generating Server Script for $Model_Name:"
# Script accepts 1 argument: path to sytorch
cat <<EOF > $SERVER_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

if [ "$1" = "clean" ]; then
  shopt -s extglob
  echo -e "\${bg_yellow}Cleaning up\${clear}"
  rm -rf !(server.sh)
  echo -e "\${bg_green}Cleaned up\${clear}"
  shopt -u extglob
  exit 0
fi

# Current directory
current_dir=\$(pwd)
echo -e "Play Area: \${bg_green}\$current_dir\${clear}"


# Clone sytorch
echo -e "\${bg_green}Cloning sytorch repository\${clear}"
git clone https://github.com/mpc-msri/EzPC
wait

sytorch="\$current_dir/EzPC/sytorch"
onnxbridge="\$current_dir/EzPC/OnnxBridge"

echo "MODEL_DIR: $MODEL_DIR"

# Copy Files to current directory
echo -e "\${bg_green}Copying files to current directory\${clear}"
cp $MODEL_PATH .
cp $PREPROCESS .

# Compile the model
echo -e "\${bg_green}Compiling the model\${clear}"
python \$onnxbridge/main.py --path $File_NAME --backend $BACKEND --scale $SCALE --bitlength $BITLENGTH --generate code
wait
\$onnxbridge/LLAMA/compile_llama.sh "${Model_Name}_${BACKEND}_${SCALE}.cpp"

# Create a zip file of stripped model to share with client
zipfile="client_$Model_Name.zip"
# Create the zip file
echo -e "\${bg_green}Creating zip file\${clear}"
zip "\$zipfile" "optimised_$File_NAME" "${Model_Name}_${BACKEND}_${SCALE}.cpp" "$preprocess_image_file"
echo -e "\${bg_green}Zip file created\${clear}"
wait

# Start a Python server to serve the stripped model
echo -e "\${bg_green}Starting a Python server to serve the stripped model to Client and Dealer.\${clear}"
python \$sytorch/scripts/server.py 2

while true; do
    # Download Keys from Dealer
    echo -e "\${bg_green}Downloading keys from Dealer\${clear}"
    # Set the Dealer IP address and port number is 9000 by default
    Dealer_url="$DEALER_IP"

    # Get the keys from the Dealer
    python \$sytorch/scripts/download_keys.py \$Dealer_url server server server.dat
    wait
    echo -e "\${bg_green}Downloaded Dealer Keys File\${clear}"

    # Model inference
    echo -e "\${bg_green}Running model inference\${clear}"
    ./${Model_Name}_${BACKEND}_${SCALE} 2 $SERVER_IP ${Model_Name}_input_weights.dat
    wait
    echo -e "\${bg_green}Model inference completed.\${clear}"
done

EOF

# Finish generating Server Script
echo "Finished generating Server Script"

# Generating Dealer Script
DEALER_SCRIPT="dealer.sh"
echo "Generating Dealer Script"
cat <<EOF > $DEALER_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

if [ "$1" = "clean" ]; then
  shopt -s extglob
  echo -e "\${bg_yellow}Cleaning up\${clear}"
  rm -rf !(dealer.sh)
  echo -e "\${bg_green}Cleaned up\${clear}"
  shopt -u extglob
  exit 0
fi

# Current directory
current_dir=\$(pwd)
echo -e "Play Area: \${bg_green}\$current_dir\${clear}"

# Downloading Server Files
echo -e "\${bg_green}Downloading Server Files\${clear}"
# Set the server IP address and port number
SERVER_PORT="8000"

# Loop until a 200 response is received
while true; do
    echo -e "\${bg_yellow}Sending GET request to server.\${clear}"
    # Send a GET request to the server and save the response status code
    STATUS=\$(curl -s -w '%{http_code}' "http://$SERVER_IP:\$SERVER_PORT/client_$Model_Name.zip" --output client_$Model_Name.zip)

    echo \$STATUS
    # Check if the status code is 200
    if [ \$STATUS -eq 200 ]; then
        echo "Zip file downloaded successfully"
        break
    fi

    echo -e "\${bg_yellow}Waiting for server to generate zip file: sleeping for 10 seconds.\${clear}"
    # Wait for 10 seconds before trying again
    sleep 10
done


# Clone sytorch
echo -e "\${bg_green}Cloning sytorch repository\${clear}"
git clone https://github.com/mpc-msri/EzPC
wait

sytorch="\$current_dir/EzPC/sytorch"
onnxbridge="\$current_dir/EzPC/OnnxBridge"

# Looking ZIP file from SERVER
echo "Looking ZIP file from SERVER"
# look for a zip file in the current directory with name "client_$Model_Name.zip"
zipfile=\$(find . -maxdepth 1 -name "client_$Model_Name.zip" -print -quit)

if [ -z "\$zipfile" ]; then
  echo "Error: Zip file not found."
  exit 1
fi

# Unzip the file
echo -e "\${bg_green}Unzipping the file\${clear}"
unzip \$zipfile 
wait

# Compile the model
echo -e "\${bg_green}Compiling the model\${clear}"
\$onnxbridge/LLAMA/compile_llama.sh "${Model_Name}_${BACKEND}_${SCALE}.cpp"
wait

# binary to generate keys
cp ${Model_Name}_${BACKEND}_${SCALE} generate_keys

# Generate keys for 1st inference
./generate_keys 1
mkdir server
mv server.dat server/server.dat
mkdir client
mv client.dat client/client.dat

# Key generation and serving key files
echo -e "\${bg_green}Starting a Python server to serve keys file\${clear}"
python \$sytorch/scripts/dealer.py $SERVER_IP 

EOF
# Finish generating Dealer Script
echo "Finished generating Dealer Script"

# Generating Client Script
CLIENT_OFFLINE_SCRIPT="client-offline.sh"
CLIENT_ONLINE_SCRIPT="client-online.sh"
echo "Generating Client Script"
# Script accepts 1 argument: path to sytorch
cat <<EOF > $CLIENT_OFFLINE_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

if [ "$1" = "clean" ]; then
  shopt -s extglob
  echo -e "\${bg_yellow}Cleaning up\${clear}"
  rm -rf !(client-o*)
  echo -e "\${bg_green}Cleaned up\${clear}"
  shopt -u extglob
  exit 0
fi

# Current directory
current_dir=\$(pwd)
echo -e "Play Area: \${bg_green}\$current_dir\${clear}"

# Downloading Server Files
echo -e "\${bg_green}Downloading Server Files\${clear}"
# Set the server IP address and port number
SERVER_PORT="8000"

# Loop until a 200 response is received
while true; do
    echo -e "\${bg_yellow}Sending GET request to server.\${clear}"
    # Send a GET request to the server and save the response status code
    STATUS=\$(curl -s -w '%{http_code}' "http://$SERVER_IP:\$SERVER_PORT/client_$Model_Name.zip" --output client_$Model_Name.zip)

    echo \$STATUS
    # Check if the status code is 200
    if [ \$STATUS -eq 200 ]; then
        echo "File downloaded successfully"
        break
    fi

    echo -e "\${bg_yellow}Waiting for server to generate zip file: sleeping for 10 seconds.\${clear}"
    # Wait for 10 seconds before trying again
    sleep 10
done
echo -e "\${bg_green}Downloaded Server Files\${clear}"

# Clone sytorch
echo -e "\${bg_green}Cloning sytorch repository\${clear}"
git clone https://github.com/mpc-msri/EzPC
wait

sytorch="\$current_dir/EzPC/sytorch"
onnxbridge="\$current_dir/EzPC/OnnxBridge"

# Looking ZIP file from SERVER
echo "Looking ZIP file from SERVER"
# look for a zip file in the current directory with name "client_$Model_Name.zip"
zipfile=\$(find . -maxdepth 1 -name "client_$Model_Name.zip" -print -quit)

if [ -z "\$zipfile" ]; then
  echo "Error: Zip file not found."
  exit 1
fi

# Unzip the file
echo -e "\${bg_green}Unzipping the file\${clear}"
unzip \$zipfile 
wait


# Compile the model
echo -e "\${bg_green}Compiling the model\${clear}"
\$onnxbridge/LLAMA/compile_llama.sh "${Model_Name}_${BACKEND}_${SCALE}.cpp"
wait



wait

EOF

cat <<EOF > $CLIENT_ONLINE_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

if [ "$1" = "clean" ]; then
  shopt -s extglob
  echo -e "\${bg_yellow}Cleaning up\${clear}"
  rm -rf !(client-o*)
  echo -e "\${bg_green}Cleaned up\${clear}"
  shopt -u extglob
  exit 0
fi

# if Image is not provided
if [ -z "\$1" ]; then
    echo "Error: Image not provided."
    exit 1
fi
IMAGE_PATH=\$1

current_dir=\$(pwd)
sytorch="\$current_dir/EzPC/sytorch"
onnxbridge="\$current_dir/EzPC/OnnxBridge"

# Download Keys from Dealer
echo -e "\${bg_green}Downloading keys from Dealer\${clear}"
# Set the dealer IP address and port number
Dealer_url="$DEALER_IP"

# Get the keys from the Dealer
python \$sytorch/scripts/download_keys.py \$Dealer_url client client client.dat
wait
echo -e "\${bg_green}Downloaded Dealer Keys File\${clear}"

# Copy the input image
cp \$IMAGE_PATH .
# get input file name
File_NAME=\$(basename \$IMAGE_PATH)
Image_Name=\${File_NAME%.*}

# Prepare the input
echo -e "\${bg_green}Preparing the input\${clear}"
python $preprocess_image_file \$File_NAME
wait
python \$onnxbridge/helper/convert_np_to_float_inp.py --inp \$Image_Name.npy --out \$Image_Name.inp


# Run the model
echo -e "\${bg_green}Running the model\${clear}"
./${Model_Name}_${BACKEND}_${SCALE} 3 $SERVER_IP < \$Image_Name.inp > output.txt

# Print the output
echo -e "\${bg_green}Printing the output\${clear}"
cat output.txt
echo -e "\${bg_green}Finished\${clear}"

EOF
echo "Finished generating Client Script"