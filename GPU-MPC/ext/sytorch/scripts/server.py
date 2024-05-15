import http.server
import socketserver
import sys

PORT = 8000
loop = int(sys.argv[1])
handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    print("Server started at localhost:" + str(PORT))
    for i in range(loop):
        httpd.handle_request()
