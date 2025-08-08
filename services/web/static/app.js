// services/web/static/app.js - FIXED VERSION
// This will work from any location (localhost, IP address, domain name)

// Dynamic WebSocket URL based on where you're accessing from
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.hostname;
// Use port 8000 for direct orchestrator connection
const wsPort = '8000';
const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws`;

console.log('Connecting to WebSocket:', wsUrl);

const ws = new WebSocket(wsUrl);
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

let currentResponse = null;

ws.onopen = () => {
    console.log('Connected to JARVIS');
    addMessage('✅ Connected to JARVIS', 'system');
};

ws.onmessage = (event) => {
    console.log('Message received:', event.data);
    
    try {
        // Try to parse as JSON first
        const data = JSON.parse(event.data);
        
        if (data.type === 'connection') {
            addMessage(data.message, 'jarvis');
        } else if (data.type === 'complete') {
            currentResponse = null;
        } else if (data.type === 'error') {
            addMessage(`❌ Error: ${data.message}`, 'system');
            currentResponse = null;
        }
    } catch (e) {
        // If not JSON, treat as plain text streaming response
        if (!currentResponse) {
            currentResponse = addMessage('', 'jarvis');
        }
        currentResponse.textContent += event.data;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    addMessage('❌ Connection error. Check if backend is running.', 'system');
};

ws.onclose = () => {
    console.log('Disconnected from JARVIS');
    addMessage('🔴 Disconnected from JARVIS. Refresh page to reconnect.', 'system');
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
    
    // Check if WebSocket is connected
    if (ws.readyState !== WebSocket.OPEN) {
        addMessage('❌ Not connected. Please refresh the page.', 'system');
        return;
    }
    
    // Display user message
    addMessage(message, 'user');
    
    // Send to backend
    const payload = JSON.stringify({
        type: 'text',
        message: message,
        use_rag: true
    });
    
    console.log('Sending message:', payload);
    ws.send(payload);
    
    // Clear input
    messageInput.value = '';
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Voice recognition setup (optional)
if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        sendMessage();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        voiceBtn.textContent = '🎤';
    };
    
    voiceBtn.addEventListener('click', () => {
        recognition.start();
        voiceBtn.textContent = '🔴';
        setTimeout(() => {
            voiceBtn.textContent = '🎤';
        }, 3000);
    });
} else {
    // Hide voice button if not supported
    if (voiceBtn) voiceBtn.style.display = 'none';
}

// Add system message class to CSS if not exists
const style = document.createElement('style');
style.textContent = `
.system-message {
    background: rgba(255, 165, 0, 0.2);
    text-align: center;
    font-style: italic;
    color: #ffa500;
}`;
document.head.appendChild(style);
