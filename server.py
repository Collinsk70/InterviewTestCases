from flask import Flask, render_template
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Key'
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for LAN

# Serve the chat page
@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('connect')
def handle_connect():
    print("New client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('message')
def handle_message(msg):
    print(f"Message received: {msg}")
    send(msg, broadcast=True)  # Send to all clients

if __name__ == '__main__':
    # Run on all interfaces so other devices can access
    socketio.run(app, host='0.0.0.0', port=5000)
