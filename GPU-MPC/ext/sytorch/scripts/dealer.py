import os
import sys
import hashlib

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import ThreadedFTPServer


class FileHandler(FTPHandler):
    files_served_to_client = 0
    files_served_to_server = 0
    keys_served = 0
    keys_available = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_connect(self):
        self.log(f"Connected {self.username}")
        self.log(f"Checking if keys are available")
        if not FileHandler.keys_available:
            self.log(f"Keys not available. Sleeping")
        while not FileHandler.keys_available:
            pass
        self.log(f"Keys available. Continuing")

    def on_file_sent(self, file):
        self.log(f"Ip of {self.username} is {self.remote_ip}")

        # Calculate the MD hash of the file
        # hasher = hashlib.md5()
        # with open(file, "rb") as f:
        #     buf = f.read()
        #     hasher.update(buf)
        # md_hash = hasher.hexdigest()

        if self.username == "server":
            FileHandler.files_served_to_server += 1
            # self.log(f"MD5 hash of server.dat is {md_hash}")
        elif self.username == "client":
            FileHandler.files_served_to_client += 1
            # self.log(f"MD5 hash of client.dat is {md_hash}")

        self.log(f"Files served to client: {FileHandler.files_served_to_client}")
        self.log(f"Files served to server: {FileHandler.files_served_to_server}")

    def on_disconnect(self):
        self.log(f"Disconnected {self.username}")
        if (
            FileHandler.files_served_to_client > 0
            and FileHandler.files_served_to_server > 0
        ):
            self.log("Files downloaded via both servers.")
            FileHandler.keys_served += 1
            FileHandler.files_served_to_client = 0
            FileHandler.files_served_to_server = 0
            self.log(f"Keys served: {FileHandler.keys_served}")

            self.log("Generating New Keys")
            FileHandler.keys_available = False
            os.system("rm -rf *.dat")
            os.system("./generate_keys 1")
            os.system("mv server.dat server/server.dat")
            os.system("mv client.dat client/client.dat")
            FileHandler.keys_available = True
            self.log("New Keys Generated")


def main():
    # Instantiate a dummy authorizer for managing 'virtual' users
    authorizer = DummyAuthorizer()
    # Define a new user having full r/w permissions and a read-only
    # anonymous user
    authorizer.add_user("server", "server", "./server", perm="elradfmwMT")
    authorizer.add_user("client", "client", "./client", perm="elradfmwMT")

    # Instantiate FTP handler class
    handler = FileHandler
    handler.authorizer = authorizer

    # Define a customized banner (string returned when client connects)
    handler.banner = "pyftpdlib based ftpd ready."

    # Instantiate FTP server class and listen on 0.0.0.0:2121
    address = (sys.argv[1], 9000)
    server = ThreadedFTPServer(address, handler)

    # set a limit for connections
    server.max_cons = 256
    server.max_cons_per_ip = 5

    # start ftp server
    server.serve_forever()


if __name__ == "__main__":
    main()
