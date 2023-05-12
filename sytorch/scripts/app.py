from flask import Flask, request, send_file
import subprocess

app = Flask(__name__)


@app.route("/inference", methods=["GET"])
def process_file():
    file = request.files["file"]
    filename = file.filename
    file.save(filename)

    subprocess.run(["./client-online.sh", "masked_image.npy"])

    # return the processed file to the user
    return send_file("output.txt", as_attachment=True)
