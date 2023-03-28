import http.server
import socketserver
import threading
import subprocess
import os

PORT_SERVER = 9000
PORT_CLIENT = 9001


class FileServer(http.server.SimpleHTTPRequestHandler):
    files_served_to_client = 0
    files_served_to_server = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)
        self.files_served_to_client = 0
        self.files_served_to_server = 0

    def end_headers(self):
        super().end_headers()
        print(f"File served to {self.server.server_address[1]}")
        if self.server.server_address[1] == PORT_SERVER:
            FileServer.files_served_to_server += 1
            print(f"Files served to server: {FileServer.files_served_to_server}")
        else:
            FileServer.files_served_to_client += 1
            print(f"Files served to client: {FileServer.files_served_to_client}")
        if (
            FileServer.files_served_to_client >= 1
            and FileServer.files_served_to_server >= 1
        ):
            # Execute the command after serving one file to each client and server
            subprocess.run(["echo", "Generating New Keys"])
            subprocess.run(["rm", "-rf", "*.dat"])
            subprocess.run(["./generate_keys", "1"])
            FileServer.files_served_to_client = 0
            FileServer.files_served_to_server = 0

    def do_GET(self):
        f = self.send_head()
        if f:
            try:
                self.send_file(f)
            except BrokenPipeError:
                # Client disconnected before the file was fully sent
                pass
            f.close()

    def send_file(self, f):
        CHUNK_SIZE = 16 * 1024
        total_size = os.fstat(f.fileno()).st_size
        bytes_sent = 0

        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            self.wfile.write(chunk)
            bytes_sent += len(chunk)
            progress = min(100, int(100 * bytes_sent / total_size))
            print(f"\rTransfer progress: [{progress:3}%] ", end="", flush=True)

        print("\rTransfer complete.                 ")


def serve_files(port, server_type):
    handler = FileServer
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"{server_type.capitalize()} started at localhost:{port}")
        while True:
            httpd.handle_request()


# Start the server and client threads
server_thread = threading.Thread(target=serve_files, args=(PORT_SERVER, "server"))
client_thread = threading.Thread(target=serve_files, args=(PORT_CLIENT, "client"))
server_thread.start()
client_thread.start()

# Loop indefinitely serving files
while True:
    pass
