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
        -i|--image)
            IMAGE_PATH="$2"
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
if [ -z "$MODEL_PATH" ] || [ -z "$IMAGE_PATH" ] || [ -z "$SERVER_IP" ] || [ -z "$PREPROCESS" ] ;
then
    echo "To run the secure MPC model, please ensure the following:"
    echo "Server Files:             Client Files:"
    echo "-------------------       --------------"
    echo "path to model.onnx        path to image.jpg"
    echo "path to preprocess.py"
    echo "server IP"
    echo "-------------------       --------------" | column -t -s $'\t'
    echo "Usage: $0 -m <full-path/model.onnx> -preprocess <full-path/preprocess_image_file> -s <server-ip> -i <full-path/image>"
    echo "Optional: [-b <backend>] [-scale <scale>] [-bl <bitlength>] "
    exit 1
fi

# Print out arguments
echo ------------------------------
echo "SERVER Details:"
echo "Model path: $MODEL_PATH"
echo "Server IP: $SERVER_IP"
echo ------------------------------
echo "CLIENT Details:"
echo "Image path: $IMAGE_PATH"
# echo "Client IP: $CLIENT_IP"
echo ------------------------------

# Getting Model Name and Directory and Model Name without extension
File_NAME=$(basename $MODEL_PATH)
MODEL_DIR=$(dirname $MODEL_PATH)
Model_Name=${File_NAME%.*}

# Get preprocess file name
preprocess_image_file=$(basename $PREPROCESS)

# Generating Server Script
SERVER_OFFLINE_SCRIPT="server-offline.sh"
SERVER_ONLINE_SCRIPT="server-online.sh"
echo "Generating Server Script for $Model_Name"
# Script accepts 1 argument: path to sytorch
cat <<EOF > $SERVER_OFFLINE_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

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
echo -e "\${bg_green}Starting a Python server to serve the stripped model\${clear}"
python \$sytorch/scripts/server.py 1

# Key generation
echo -e "\${bg_green}Generating keys\${clear}"
./${Model_Name}_${BACKEND}_${SCALE} 1
rm client.dat

EOF

cat <<EOF > $SERVER_ONLINE_SCRIPT
#!/bin/bash

# Color variables
bg_red='\033[0;41m'
bg_green='\033[0;42m'
bg_yellow='\033[0;43m'
bg_blue='\033[0;44m'
bg_magenta='\033[0;45m'
bg_cyan='\033[0;46m'
clear='\033[0m'

# Model inference
echo -e "\${bg_green}Running model inference\${clear}"
./${Model_Name}_${BACKEND}_${SCALE} 2 $SERVER_IP ${Model_Name}_input_weights.dat
echo -e "\${bg_green}Model inference completed.\${clear}"

EOF

# Finish generating Server Script
echo "Finished generating Server Script"



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

# Generate the key
echo -e "\${bg_green}Generating the key\${clear}"
./${Model_Name}_${BACKEND}_${SCALE} 1
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

current_dir=\$(pwd)
sytorch="\$current_dir/EzPC/sytorch"
onnxbridge="\$current_dir/EzPC/OnnxBridge"

# Copy the input image
cp $IMAGE_PATH .
# get input file name
File_NAME=\$(basename $IMAGE_PATH)
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