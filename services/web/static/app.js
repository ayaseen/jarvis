// services/web/static/app.js
const ws = new WebSocket('ws://localhost:8000/ws');
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

let currentResponse = null;

ws.onopen = () => {
    console.log('Connected to JARVIS');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'connection') {
        addMessage(data.message, 'jarvis');
    } else if (data.type === 'complete') {
        currentResponse = null;
    } else {
        if (!currentResponse) {
            currentResponse = addMessage('', 'jarvis');
        }
        currentResponse.textContent += data;
    }
};

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return messageDiv;
}

function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    addMessage(message, 'user');
    
    ws.send(JSON.stringify({
        type: 'text',
        message: message,
        use_rag: true
    }));
    
    messageInput.value = '';
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Voice recognition setup
if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        sendMessage();
    };
    
    voiceBtn.addEventListener('click', () => {
        recognition.start();
        voiceBtn.textContent = '🔴';
        setTimeout(() => {
            voiceBtn.textContent = '🎤';
        }, 3000);
    });
}
