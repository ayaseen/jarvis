// services/web/static/app.js
// WebSocket through nginx proxy - NOT direct to 8000!

// Use nginx proxy path, not direct connection
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;  // This goes through nginx!

console.log('JARVIS WebSocket URL:', wsUrl);

// Global variables
let ws = null;
let currentResponse = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
let reconnectTimeout = null;
let heartbeatInterval = null;
let isConnecting = false;
let sessionId = null;

// DOM elements - USING CORRECT IDs!
const messagesDiv = document.getElementById('messages');  // NOT chat-messages!
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
    console.log('Initializing WebSocket connection to:', wsUrl);
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket Connected to JARVIS');
            isConnecting = false;
            reconnectAttempts = 0;
            clearSystemMessages();
            
            // Update connection status
            updateConnectionStatus(true);
            
            // Enable input
            if (messageInput) messageInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            
            // Start heartbeat
            startHeartbeat();
        };
        
        ws.onmessage = (event) => {
            console.log('Message received:', event.data);
            handleMessage(event.data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            isConnecting = false;
            addMessage('❌ Connection error. Check if backend is running.', 'system');
            updateConnectionStatus(false);
            
            // Disable input
            if (messageInput) messageInput.disabled = true;
            if (sendBtn) sendBtn.disabled = true;
            
            // Try to reconnect
            scheduleReconnect();
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket closed. Code:', event.code, 'Reason:', event.reason);
            isConnecting = false;
            ws = null;
            
            // Stop heartbeat
            stopHeartbeat();
            updateConnectionStatus(false);
            
            // Disable input
            if (messageInput) messageInput.disabled = true;
            if (sendBtn) sendBtn.disabled = true;
            
            if (event.code !== 1000) { // 1000 = normal closure
                addMessage('🔴 Disconnected from JARVIS. Reconnecting...', 'system');
                scheduleReconnect();
            }
        };
    } catch (error) {
        console.error('Failed to create WebSocket:', error);
        isConnecting = false;
        addMessage('❌ Failed to connect. Please check services.', 'system');
        updateConnectionStatus(false);
    }
}

// Handle incoming messages
function handleMessage(data) {
    try {
        // Try to parse as JSON first
        const jsonData = JSON.parse(data);
        console.log('Parsed message:', jsonData);
        
        if (jsonData.type === 'connection') {
            // Connection established message
            sessionId = jsonData.session_id;
            console.log('Session established:', sessionId);
            // Show welcome message
            if (jsonData.message) {
                addMessage(jsonData.message, 'jarvis');
            }
        } else if (jsonData.type === 'processing') {
            // Show processing message
            if (jsonData.message) {
                addMessage(jsonData.message, 'system');
            }
        } else if (jsonData.type === 'chunk') {
            // Streaming chunk
            if (!currentResponse) {
                currentResponse = addMessage('', 'jarvis');
            }
            if (currentResponse && jsonData.content) {
                currentResponse.textContent += jsonData.content;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        } else if (jsonData.type === 'complete') {
            // Response complete
            if (jsonData.message && currentResponse) {
                currentResponse.textContent = jsonData.message;
            } else if (jsonData.message) {
                addMessage(jsonData.message, 'jarvis');
            }
            currentResponse = null;
            console.log('Response complete');
        } else if (jsonData.type === 'error') {
            // Error message
            addMessage(`❌ Error: ${jsonData.message}`, 'system');
            currentResponse = null;
        }
    } catch (e) {
        // If not JSON, treat as plain text response
        console.log('Plain text message:', data);
        if (!currentResponse) {
            currentResponse = addMessage('', 'jarvis');
        }
        if (currentResponse) {
            currentResponse.textContent += data;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    }
}

// Add message to chat
function addMessage(text, sender) {
    if (!messagesDiv) {
        console.error('Messages div not found!');
        return null;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return messageDiv;
}

// Clear system messages
function clearSystemMessages() {
    if (!messagesDiv) return;
    
    const systemMessages = messagesDiv.querySelectorAll('.system-message');
    systemMessages.forEach(msg => {
        if (msg.textContent.includes('Disconnected') || 
            msg.textContent.includes('Reconnecting') ||
            msg.textContent.includes('Connecting')) {
            msg.remove();
        }
    });
}

// Update connection status
function updateConnectionStatus(connected) {
    const statusElement = document.querySelector('.connection-status') || 
                         document.getElementById('connection-status');
    
    if (statusElement) {
        if (connected) {
            statusElement.textContent = '● Connected';
            statusElement.style.color = '#00ff00';
        } else {
            statusElement.textContent = '● Connection error';
            statusElement.style.color = '#ff0000';
        }
    }
}

// Send message
function sendMessage() {
    if (!messageInput) return;
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Check if WebSocket is connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addMessage('❌ Not connected. Please wait...', 'system');
        initWebSocket();
        return;
    }
    
    // Display user message
    addMessage(message, 'user');
    
    // Send to backend
    const payload = {
        message: message,
        use_rag: document.getElementById('use-rag')?.checked ?? true,
        session_id: sessionId
    };
    
    console.log('Sending message:', payload);
    
    try {
        ws.send(JSON.stringify(payload));
        // Clear input
        messageInput.value = '';
        messageInput.focus();
    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('❌ Failed to send message.', 'system');
    }
}

// Heartbeat mechanism
function startHeartbeat() {
    stopHeartbeat(); // Clear any existing interval
    
    heartbeatInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log('Sending heartbeat');
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
        addMessage('❌ Max reconnection attempts reached. Please refresh.', 'system');
        return;
    }
    
    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000);
    
    console.log(`Reconnect attempt ${reconnectAttempts} in ${delay}ms`);
    
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
    }
    
    reconnectTimeout = setTimeout(() => {
        addMessage(`🔄 Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`, 'system');
        initWebSocket();
    }, delay);
}

// Event listeners
if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
}

if (messageInput) {
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// Voice button handler
if (voiceBtn) {
    voiceBtn.addEventListener('click', () => {
        addMessage('🎤 Voice input not yet implemented', 'system');
    });
}

// Add styles
const style = document.createElement('style');
style.textContent = `
.message {
    padding: 10px;
    margin: 5px 0;
    border-radius: 10px;
    animation: fadeIn 0.3s;
}

.user-message {
    background: #667eea;
    color: white;
    margin-left: 20%;
    text-align: right;
}

.jarvis-message {
    background: #48bb78;
    color: white;
    margin-right: 20%;
}

.system-message {
    background: rgba(255, 193, 7, 0.2);
    color: #856404;
    text-align: center;
    font-style: italic;
    font-size: 0.9em;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.connection-status {
    position: fixed;
    top: 10px;
    right: 150px;
    font-weight: bold;
}
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('JARVIS Web Interface Loaded');
    
    // Add connection status indicator if not exists
    if (!document.querySelector('.connection-status')) {
        const statusDiv = document.createElement('div');
        statusDiv.className = 'connection-status';
        statusDiv.id = 'connection-status';
        statusDiv.textContent = '● Connecting...';
        statusDiv.style.color = '#ffa500';
        document.body.appendChild(statusDiv);
    }
    
    // Initial connection message
    addMessage('🔄 Connecting to JARVIS...', 'system');
    
    // Start WebSocket connection
    initWebSocket();
    
    // Handle visibility changes
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && (!ws || ws.readyState !== WebSocket.OPEN)) {
            console.log('Page visible, checking connection...');
            initWebSocket();
        }
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
