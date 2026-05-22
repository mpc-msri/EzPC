from flask import Flask, request, send_file
import subprocess
import threading
import os
import signal

app = Flask(__name__)
shutdown_event = threading.Event()


@app.route("/inference", methods=["GET"])
def process_file():
    file = request.files["file"]
    filename = file.filename
    file.save(filename)

    subprocess.run(["./client-online.sh", "masked_image.npy"])

    # return the processed file to the user
    response = send_file("output.txt", as_attachment=True)

    # trigger server shutdown in a background thread
    shutdown_thread = threading.Thread(target=shutdown_server)
    shutdown_thread.start()

    return response


def shutdown_server():
    # Wait for a brief period to allow the ongoing request to complete
    shutdown_event.wait(timeout=2)

    # Stop the Flask server by terminating the Python process
    os.kill(os.getpid(), signal.SIGINT)
