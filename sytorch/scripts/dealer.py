import os
import sys

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import ThreadedFTPServer


class FileHandler(FTPHandler):
    files_served_to_client = 0
    files_served_to_server = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_file_sent(self, file):
        self.log(f"Ip of {self.username} is {self.remote_ip}")
        if self.username == "server":
            FileHandler.files_served_to_server += 1
        elif self.username == "client":
            FileHandler.files_served_to_client += 1
        self.log(f"Files served to client: {FileHandler.files_served_to_client}")
        self.log(f"Files served to server: {FileHandler.files_served_to_server}")
        if (
            FileHandler.files_served_to_client > 0
            and FileHandler.files_served_to_server > 0
        ):
            self.log("Files downloaded via both servers.")
            FileHandler.files_served_to_client = 0
            FileHandler.files_served_to_server = 0
            self.log("Generating New Keys")
            os.system("rm -rf *.dat")
            os.system("./generate_keys 1")
            os.system("mv server.dat server/server.dat")
            os.system("mv client.dat client/client.dat")
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
