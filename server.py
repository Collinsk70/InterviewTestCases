from flask import Flask, render_template, request
from flask_socketio import SocketIO, send, emit
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

users = {}  # Maps socket_id -> username

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('set_username')
def handle_set_username(username):
    users[request.sid] = username
    print(f"User connected: {username}")
    emit("user_list", list(users.values()), broadcast=True)

@socketio.on('message')
def handle_message(data):
    # Data: { user: <username>, text: <encrypted>, timestamp: <client_time> }
    print(f"[{data['timestamp']}] {data['user']} sent an encrypted message")
    send(data, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    username = users.pop(request.sid, "Unknown")
    print(f"User disconnected: {username}")
    emit("user_list", list(users.values()), broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
