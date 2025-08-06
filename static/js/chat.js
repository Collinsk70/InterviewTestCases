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
