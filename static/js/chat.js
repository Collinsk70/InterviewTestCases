const SERVER_IP = ""; // Replace with your IP Address
const socket = io(`http://${SERVER_IP}:5000`);
const SECRET_KEY = "mysecretkey123";

let username = "";

function login() {
  const input = document.getElementById('username');
  if (input.value.trim() !== "") {
    username = input.value.trim();
    document.getElementById('login').style.display = 'none';
    document.getElementById('chat-layout').style.display = 'flex';
    socket.emit('set_username', username);
  }
}

function encryptMessage(message) {
  return CryptoJS.AES.encrypt(message, SECRET_KEY).toString();
}

function decryptMessage(encrypted) {
  try {
    const bytes = CryptoJS.AES.decrypt(encrypted, SECRET_KEY);
    return bytes.toString(CryptoJS.enc.Utf8);
  } catch (e) {
    return "[Decryption Error]";
  }
}

socket.on('message', data => {
  const { user, text, timestamp } = data;
  const msg = decryptMessage(text);
  addMessage(user, msg, timestamp);
});

socket.on('user_list', users => {
  const list = document.getElementById('user-list');
  list.innerHTML = "";
  users.forEach(u => {
    const li = document.createElement('li');
    li.textContent = u;
    list.appendChild(li);
  });
});

function addMessage(user, msg, timestamp) {
  const li = document.createElement('li');
  li.classList.add('message', user === username ? 'me' : 'other');
  li.innerHTML = `<b>${user}</b><br>${msg}<span class="time">${timestamp}</span>`;
  document.getElementById('messages').appendChild(li);
  li.scrollIntoView({ behavior: "smooth" });
}

function sendMessage() {
  const input = document.getElementById('message');
  const msg = input.value.trim();
  if (msg !== "") {
    const encrypted = encryptMessage(msg);
    const timestamp = new Date().toLocaleTimeString();
    socket.send({ user: username, text: encrypted, timestamp });
    input.value = "";
  }
}
let localStream;
let peerConnection;
const peerConfig = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };

const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");

function startCall() {
  navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then(stream => {
    localStream = stream;
    localVideo.srcObject = stream;

    peerConnection = new RTCPeerConnection(peerConfig);
    stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

    peerConnection.onicecandidate = event => {
      if (event.candidate) {
        socket.emit('ice-candidate', { to: 'all', candidate: event.candidate });
      }
    };

    peerConnection.ontrack = event => {
      remoteVideo.srcObject = event.streams[0];
    };

    peerConnection.createOffer().then(offer => {
      peerConnection.setLocalDescription(offer);
      socket.emit('video-offer', { sdp: offer });
    });
  });
}

socket.on('video-offer', (data) => {
  navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then(stream => {
    localStream = stream;
    localVideo.srcObject = stream;

    peerConnection = new RTCPeerConnection(peerConfig);
    stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

    peerConnection.onicecandidate = event => {
      if (event.candidate) {
        socket.emit('ice-candidate', { candidate: event.candidate });
      }
    };

    peerConnection.ontrack = event => {
      remoteVideo.srcObject = event.streams[0];
    };

    peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp)).then(() => {
      return peerConnection.createAnswer();
    }).then(answer => {
      return peerConnection.setLocalDescription(answer);
    }).then(() => {
      socket.emit('video-answer', { sdp: peerConnection.localDescription });
    });
  });
});

socket.on('video-answer', (data) => {
  peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
});

socket.on('ice-candidate', (data) => {
  if (data.candidate) {
    peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
  }
});
