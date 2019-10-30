import http.server
import socketserver 
import socket
import sys

class CustomTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def close(self):
        self.socket.close()

httpd = CustomTCPServer(("0.0.0.0", int(sys.argv[1])), http.server.SimpleHTTPRequestHandler)
    

max_requests = 1
i=1

while i <= max_requests:
    httpd.handle_request()
    i += 1

httpd.close()
print("Sent info to the client")
