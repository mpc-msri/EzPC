from threading import Thread
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from pyftpdlib.authorizers import DummyAuthorizer
import subprocess, os
import logging

PORT_SERVER = 9005
PORT_CLIENT = 9006


class MyHandler(FTPHandler):
    files_served_to_client = 0
    files_served_to_server = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_served_to_client = 0
        self.files_served_to_server = 0

    def on_file_sent(self, file):
        # Increment the appropriate download counter
        if self.server_address[1] == PORT_SERVER:
            self.files_served_to_server += 1
        elif self.server_address[1] == PORT_CLIENT:
            self.files_served_to_client += 1

        # Check if files have been downloaded via both servers
        if self.files_served_to_client > 0 and self.files_served_to_server > 0:
            # Run the desired commands here
            subprocess.run(["echo", "Generating New Keys"])
            subprocess.run(["rm", "-rf", "*.dat"])
            subprocess.run(["./generate_keys", "1"])
            print("Files downloaded via both servers.")
            self.files_served_to_client = 0
            self.files_served_to_server = 0


logging.basicConfig(level=logging.DEBUG)

cwd = os.getcwd()
authorizer1 = DummyAuthorizer()
authorizer1.add_user("user", "pass", ".", perm="elradfmwMT")
authorizer1.add_anonymous(os.getcwd())
handler1 = MyHandler
handler1.authorizer = authorizer1
# Create two instances of the FTP server
server1 = FTPServer(("0.0.0.0", PORT_SERVER), handler1)

authorizer2 = DummyAuthorizer()
authorizer2.add_user("client", "client", ".", perm="elradfmwMT")
authorizer2.add_anonymous(os.getcwd())
handler2 = MyHandler
handler2.authorizer = authorizer2
server2 = FTPServer(("0.0.0.0", PORT_CLIENT), handler2)

# Define a function to start the servers in separate threads
def start_servers():
    server1.serve_forever()


def start_servers2():
    server2.serve_forever()


# Start the servers in separate threads
Thread(target=start_servers).start()
Thread(target=start_servers2).start()
