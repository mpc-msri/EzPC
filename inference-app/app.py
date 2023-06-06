import gradio as gr
import time, os
from PIL import Image
import numpy as np
import ftplib
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from constants import (
    desc,
    Input_Shape,
    EXAMPLES,
    preprocess,
    dims,
    scale,
    mode,
    labels_map,
)

url = os.getenv("_URL")
user = os.getenv("_USER")
passwd = os.getenv("_PASSWORD")
file_name = os.getenv("_FILE_NAME")
client_ip = os.getenv("_CLIENT_IP")


print("Starting the demo...")
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        f"""
        <!-- =====BOX ICONS===== -->
        <link href='https://cdn.jsdelivr.net/npm/boxicons@2.0.5/css/boxicons.min.css' rel='stylesheet'>
        <link href='https://unpkg.com/boxicons@2.0.7/css/boxicons.min.css' rel='stylesheet'>


        <h1 align="center" >
        <center>
            <img src="file/Assets/onnxBridge.jpg" alt="EzPC" width="20%" height="20%"  />
        </center>
        </h1>
        <h1 align="center">Securely Inferencing a Machine Learning model using EzPC</h1>

        
        <p align="center">
            <a href="https://github.com/mpc-msri/EzPC"> <i class='bx bxl-github' ></i> EzPC</a>
            —
            <a href="https://www.microsoft.com/en-us/research/project/ezpc-easy-secure-multi-party-computation/"> <i class='bx bxl-microsoft'></i> Project</a>
        </p>
        <br><br>

        <p align="center" >
            <img src="file/Assets/computation.png" width="70%" height="70%">
        </p>

        <p align="center">
            {desc}
        </p>

        <p align="center">
            Try out the below app, and see
            <a href="https://github.com/mpc-msri/EzPC/tree/master/OnnxBridge"> tutorial</a>
            for more info!
        </p>
        """
    )

    gr.Markdown("## Client side")

    # Step 1 Input Image
    gr.Markdown("### Step 1: Upload an image. ")
    gr.Markdown(
        f"The image will automatically be resized to shape {Input_Shape} as the input size for lenet model. "
    )

    with gr.Row():
        input_image = gr.Image(
            value=None,
            label="Upload an image here.",
            shape=(dims["h"], dims["w"]),
            source="upload",
            interactive=True,
            image_mode=mode,
            type="pil",
        )
        examples = gr.Examples(
            examples=EXAMPLES,
            inputs=[input_image],
            examples_per_page=5,
            label="Examples to use.",
        )

    # Step 2 Get Mask from Dealer
    gr.Markdown(
        "### Step 2: Click on the button below to get encryption keys from dealer."
    )
    dealer_status = gr.Textbox(
        label="Status", placeholder="Encryption Keys status will be shown here."
    )
    get_mask_button = gr.Button(value="Get Encryption Keys", interactive=True)

    # Step 3 Mask Input Image
    gr.Markdown("### Step 3: Click on the button below to encrypt the image.")
    with gr.Row():
        in_image = gr.Image(
            value=None,
            label="Input Image",
            shape=(dims["h"], dims["w"]),
            interactive=False,
            image_mode=mode,
            type="pil",
        ).style(width=256, height=256)
        out_image = gr.Image(
            value=None,
            label="Encrypted Image",
            shape=(dims["h"], dims["w"]),
            interactive=False,
        ).style(width=256, height=256)
    mask_button = gr.Button(value="Encrypt Image", interactive=True)

    # Step 4 Start Secure Inference
    gr.Markdown(
        "### Step 4: Click on the button below to start secure inference with Encrypted Image."
    )
    with gr.Column():
        inference_status = gr.Textbox(
            show_label=False,
            placeholder="Inference status will be shown here.",
            interactive=False,
        )
        inference_button = gr.Button(value="Start Secure Inference", interactive=True)
        prediction = gr.Label("Prediction: ", interactive=False, visible=False)

    def show_progress(progress=gr.Progress()):
        for i in range(10):
            time.sleep(0.1)
            progress(i / 10, desc="Encrypting Image")
        return True

    def update_input_image(input_image):
        return input_image

    def check_dealer_status(progress=gr.Progress()):
        try:
            progress(0.001, desc="Connecting with Dealer\n Please wait...")
            progress(0.035, desc="Dealer is still generating keys\n Please wait...")
            ftp = ftplib.FTP()
            print(f"Connecting to {url}")
            ftp.connect(url, 9000)
            progress(0.05, desc="Authenticating with Dealer")
            ftp.login(user=user, passwd=passwd)
            progress(0.1, desc="Authenticated Successfully")

            # Switch to binary mode
            ftp.sendcmd("TYPE i")

            # Get the size of the file on the server
            file_size = ftp.size(file_name)
            print(f"File size: {file_size}")

            xbar = 0.1
            # Download the file and display a progress bar
            with open(file_name, "wb") as f:
                with tqdm(
                    unit="B", unit_scale=True, unit_divisor=1024, total=file_size
                ) as pbar:

                    def callback(data):
                        f.write(data)
                        pbar.update(len(data))
                        progress(
                            xbar + (1 - xbar) * pbar.n / file_size,
                            desc="Downloading Encryption Keys",
                        )

                    ftp.retrbinary(f"RETR {file_name}", callback)

            ftp.quit()
            return {
                dealer_status: gr.update(
                    value="Encryption Keys received from dealer.", visible=True
                )
            }

        except Exception as e:
            print(f"Error: {e}")
            # print(f"Error: Dealer not ready.")
            return {
                dealer_status: gr.update(
                    value="Dealer not ready, please try again after some time.",
                    visible=True,
                )
            }

    def mask_image(input_image, progress=gr.Progress()):
        arr = preprocess(input_image)

        # Open the file for reading
        with open("masks.dat", "r") as f:
            # Read the contents of the file as a list of integers
            data = [int(line.strip()) for line in f.readlines()]

        np_mask = np.array(data).reshape((1, dims["h"], dims["w"], dims["c"]))
        np_mask = np.transpose(np_mask, (0, 3, 1, 2))

        print("Masking Image")
        arr_save = arr.copy()
        arr_save = arr_save * (1 << scale)
        arr_save = arr_save.astype(np.int64)
        arr_save = arr_save + np_mask
        np.save("masked_image.npy", arr_save)

        # for debugging
        # with open("masked_inp.inp", "w") as f:
        #     for x in np.nditer(arr_save, order='C'):
        #         f.write(str(x) + "\n")

        if mode == "RGB":
            arr_save = arr_save.reshape(Input_Shape[1:])
            print(arr_save.shape)
            arr_converted = np.transpose(arr_save, (1, 2, 0))
            updated_image = Image.fromarray(arr_converted, mode=mode)
            show_progress(progress)
            print(updated_image.size)
        elif mode == "L":
            arr_save = arr_save.reshape(Input_Shape[2:])
            print(arr_save.shape)
            updated_image = Image.fromarray(arr_save, mode=mode)
            show_progress(progress)
            print(updated_image.size)
        else:
            print("Invalid Mode")
            return None
        return updated_image

    def start_inference(in_image, progress=gr.Progress()):
        print("Starting Inference")
        url = f"http://{client_ip}:5000/inference"
        file_path = "masked_image.npy"
        with open(file_path, "rb") as file:
            try:
                response = requests.get(url, files={"file": file})

                if response.status_code == 200:
                    with open("output.txt", "wb") as file:
                        print(response.content)
                        file.write(response.content)
                else:
                    print("Error:", response.status_code)
                    return {
                        inference_status: gr.update(
                            value=f"Error {response.status_code} \nClient in Setup Phase...",
                            visible=True,
                        )
                    }

            except requests.Timeout:
                print("Connection timeout.")
                return {
                    prediction: gr.update(
                        value=f"Connection Timedout. \nClient in Setup Phase...",
                        visible=True,
                    )
                }
            except requests.HTTPError as e:
                print("HTTP Error:", e)
                return {
                    prediction: gr.update(
                        value=f"Error {e} \nClient in Setup Phase...", visible=True
                    )
                }
            except requests.RequestException as e:
                print("Error:", e)
                return {
                    prediction: gr.update(
                        value=f"Connection Refused. \nClient in Setup Phase...",
                        visible=True,
                    )
                }

        # Read the contents of the file as a list of integers and return the index of max value
        with open("output.txt", "r") as f:
            # Read the contents of the file as a list of integers
            data_as_str = [line.strip() for line in f.readlines()]
            data = data_as_str[0].split(" ")
            data = [float(i) for i in data]
            # find the index of max value
            print(data)
            print(type(data))
            print(type(data[0]))
            index = data.index(max(data))
            print(f"Prediction: {labels_map[index]}")
        return {
            prediction: gr.update(
                value=f"Prediction: {labels_map[index]}", visible=True
            )
        }

    get_mask_button.click(fn=check_dealer_status, inputs=[], outputs=[dealer_status])

    mask_button.click(fn=mask_image, inputs=[input_image], outputs=[out_image])

    inference_button.click(
        fn=start_inference, inputs=[out_image], outputs=[inference_status, prediction]
    )

    input_image.change(fn=update_input_image, inputs=[input_image], outputs=[in_image])

demo.queue(concurrency_count=20).launch(share=False)
