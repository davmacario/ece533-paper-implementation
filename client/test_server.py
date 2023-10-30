import http.server
import socketserver
import json
from curvefittingnn import CurveFitter, targetFunction

port = 5555
n_data = 1000

# as the central we want to create a global model
global_model = CurveFitter(24, targetFunction)

# Create a custom request handler that allows POST requests
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
    # handle POST requests here
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"Received POST data: {post_data}".encode('utf-8'))
    def do_GET(self):
        if self.path == "/data":
            self.send_response(200)
            print("client requesting data")
            # for every request, send n_data new random data points
            t_x, t_y = global_model.createTrainSet(n_data)
            data_to_send = {"t_x": t_x.tolist(), "t_y": t_y.tolist()}
            with open('test_data.json', 'w') as new_json:
                json.dump(data_to_send, new_json)

            with open('test_data.json', "rb") as datafile:
                try:
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(datafile.read())
                except:
                    print("error sending json")
        self.end_headers()

# Create the server
with socketserver.TCPServer(("127.0.0.1", port), CustomHandler) as httpd:
    print(f"Server started on port {port}")
    httpd.serve_forever()
