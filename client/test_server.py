import http.server
import socketserver

# Set the port you want to use
port = 5555

# Create a custom request handler that allows POST requests
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
    # You can handle POST requests here
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"Received POST data: {post_data}".encode('utf-8'))

# Create the server
with socketserver.TCPServer(("127.0.0.1", port), CustomHandler) as httpd:
    print(f"Server started on port {port}")
    httpd.serve_forever()
