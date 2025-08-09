// services/web/static/app.js
// Complete WebSocket client with error handling and reconnection

// Dynamic WebSocket URL based on where you're accessing from
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.hostname;
const wsPort = '8000';
const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws`;

console.log('JARVIS WebSocket URL:', wsUrl);

// Global variables
let ws = null;
let currentResponse = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
let reconnectTimeout = null;
let heartbeatInterval = null;
let isConnecting = false;

// DOM elements
const messagesDiv = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

// Initialize WebSocket connection
function initWebSocket() {
    if (isConnecting || (ws && ws.readyState === WebSocket.OPEN)) {
        console.log('WebSocket already connected or connecting');
        return;
    }
    
    isConnecting = true;
    console.log('Initializing WebSocket connection...');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('Connected to JARVIS');
        isConnecting = false;
        reconnectAttempts = 0;
        clearSystemMessages();
        addMessage('✅ Connected to JARVIS', 'system');
        
        // Enable input
        messageInput.disabled = false;
        sendBtn.disabled = false;
        
        // Start heartbeat
        startHeartbeat();
    };
    
    ws.onmessage = (event) => {
        handleMessage(event.data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnecting = false;
        addMessage('❌ Connection error. Check if backend is running.', 'system');
        
        // Disable input
        messageInput.disabled = true;
        sendBtn.disabled = true;
        
        // Try to reconnect
        scheduleReconnect();
    };
    
    ws.onclose = (event) => {
        console.log('Disconnected from JARVIS. Code:', event.code, 'Reason:', event.reason);
        isConnecting = false;
        
        // Stop heartbeat
        stopHeartbeat();
        
        // Disable input
        messageInput.disabled = true;
        sendBtn.disabled = true;
        
        if (event.code !== 1000) { // 1000 = normal closure
            addMessage('🔴 Disconnected from JARVIS. Attempting to reconnect...', 'system');
            scheduleReconnect();
        }
    };
}

// Handle incoming messages
function handleMessage(data) {
    try {
        // Try to parse as JSON first
        const jsonData = JSON.parse(data);
        
        if (jsonData.type === 'connection') {
            // Connection established message
            console.log('Connection confirmed:', jsonData.session_id);
            if (jsonData.message && jsonData.message !== "JARVIS online. How can I assist you?") {
                addMessage(jsonData.message, 'jarvis');
            }
        } else if (jsonData.type === 'complete') {
            // Response complete
            currentResponse = null;
            console.log('Response complete');
        } else if (jsonData.type === 'error') {
            // Error message
            addMessage(`❌ Error: ${jsonData.message}`, 'system');
            currentResponse = null;
        } else if (jsonData.type === 'timeout') {
            // Timeout message
            addMessage(`⏱️ ${jsonData.message}`, 'system');
        } else if (jsonData.type === 'pong') {
            // Heartbeat response
            console.log('Heartbeat pong received');
        } else if (jsonData.type === 'transcription') {
            // Voice transcription
            addMessage(`🎤 Transcribed: ${jsonData.text}`, 'system');
        }
    } catch (e) {
        // If not JSON, treat as streaming text response
        if (!currentResponse) {
            currentResponse = addMessage('', 'jarvis');
        }
        currentResponse.textContent += data;
        // Auto-scroll to bottom
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
}

// Add message to chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return messageDiv;
}

// Clear system messages
function clearSystemMessages() {
    const systemMessages = messagesDiv.querySelectorAll('.system-message');
    systemMessages.forEach(msg => {
        if (msg.textContent.includes('Disconnected') || msg.textContent.includes('Reconnecting')) {
            msg.remove();
        }
    });
}

// Send message
function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Check if WebSocket is connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addMessage('❌ Not connected. Please wait for reconnection...', 'system');
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
    
    try {
        ws.send(payload);
        // Clear input
        messageInput.value = '';
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('❌ Failed to send message. Please try again.', 'system');
    }
}

// Heartbeat mechanism
function startHeartbeat() {
    stopHeartbeat(); // Clear any existing interval
    
    heartbeatInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log('Sending heartbeat ping');
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000); // Send ping every 30 seconds
}

function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }
}

// Reconnection logic
function scheduleReconnect() {
    if (reconnectAttempts >= maxReconnectAttempts) {
        addMessage('❌ Maximum reconnection attempts reached. Please refresh the page.', 'system');
        return;
    }
    
    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000); // Exponential backoff
    
    console.log(`Scheduling reconnect attempt ${reconnectAttempts} in ${delay}ms`);
    
    reconnectTimeout = setTimeout(() => {
        addMessage(`🔄 Reconnection attempt ${reconnectAttempts}/${maxReconnectAttempts}...`, 'system');
        initWebSocket();
    }, delay);
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Voice recognition setup (optional)
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        console.log('Voice recognition started');
        voiceBtn.textContent = '🔴';
        voiceBtn.classList.add('recording');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('Voice transcript:', transcript);
        messageInput.value = transcript;
        sendMessage();
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        voiceBtn.textContent = '🎤';
        voiceBtn.classList.remove('recording');
        
        if (event.error === 'no-speech') {
            addMessage('⚠️ No speech detected. Please try again.', 'system');
        } else if (event.error === 'not-allowed') {
            addMessage('❌ Microphone access denied. Please allow microphone access.', 'system');
        } else {
            addMessage(`❌ Voice recognition error: ${event.error}`, 'system');
        }
    };
    
    recognition.onend = () => {
        console.log('Voice recognition ended');
        voiceBtn.textContent = '🎤';
        voiceBtn.classList.remove('recording');
    };
    
    voiceBtn.addEventListener('click', () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            addMessage('❌ Not connected. Voice input unavailable.', 'system');
            return;
        }
        
        try {
            recognition.start();
        } catch (error) {
            console.error('Failed to start recognition:', error);
            addMessage('❌ Failed to start voice recognition.', 'system');
        }
    });
} else {
    // Hide voice button if not supported
    if (voiceBtn) {
        voiceBtn.style.display = 'none';
        console.log('Voice recognition not supported in this browser');
    }
}

// Add custom styles
const style = document.createElement('style');
style.textContent = `
.system-message {
    background: rgba(255, 165, 0, 0.2);
    text-align: center;
    font-style: italic;
    color: #ffa500;
    padding: 10px;
    margin: 10px 0;
    border-radius: 10px;
}

.message {
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { 
        opacity: 0; 
        transform: translateY(10px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

#message-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.recording {
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

#messages {
    scroll-behavior: smooth;
}

.jarvis-message {
    white-space: pre-wrap;
    word-wrap: break-word;
}
`;
document.head.appendChild(style);

// Initialize connection on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded, initializing JARVIS...');
    
    // Initial connection message
    addMessage('🔄 Connecting to JARVIS...', 'system');
    
    // Start WebSocket connection
    initWebSocket();
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && (!ws || ws.readyState !== WebSocket.OPEN)) {
            console.log('Page became visible, checking connection...');
            initWebSocket();
        }
    });
    
    // Handle online/offline events
    window.addEventListener('online', () => {
        console.log('Network online, checking connection...');
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            initWebSocket();
        }
    });
    
    window.addEventListener('offline', () => {
        console.log('Network offline');
        addMessage('❌ Network connection lost', 'system');
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopHeartbeat();
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Page unload');
    }
});
