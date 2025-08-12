// services/web/static/chat.js

// Global variables
let ws = null;
let currentConversationId = null;
let currentSessionId = null;
let conversations = [];
let isConnected = false;
let reconnectAttempts = 0;
let useRAG = true;
let currentResponse = null;
let isGenerating = false;

// WebSocket configuration
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('JARVIS Chat Interface Loaded');
    
    // Load settings
    loadSettings();
    
    // Initialize WebSocket
    initWebSocket();
    
    // Load conversations
    loadConversations();
    
    // Setup auto-resize for textarea
    setupTextareaAutoResize();
    
    // Check for conversation ID in URL
    const urlParams = new URLSearchParams(window.location.search);
    const conversationId = urlParams.get('conversation');
    if (conversationId) {
        loadConversation(conversationId);
    }
    
    // Enable/disable send button based on input
    const messageInput = document.getElementById('messageInput');
    messageInput.addEventListener('input', () => {
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = messageInput.value.trim() === '' || isGenerating;
    });
});

// Initialize WebSocket connection
function initWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected');
        return;
    }
    
    console.log('Connecting to WebSocket:', wsUrl);
    updateConnectionStatus('connecting');
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket Connected');
            isConnected = true;
            reconnectAttempts = 0;
            updateConnectionStatus('connected');
            
            // Start heartbeat
            startHeartbeat();
        };
        
        ws.onmessage = (event) => {
            handleWebSocketMessage(event.data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus('error');
        };
        
        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            isConnected = false;
            updateConnectionStatus('disconnected');
            
            // Attempt to reconnect
            if (reconnectAttempts < 5) {
                setTimeout(() => {
                    reconnectAttempts++;
                    initWebSocket();
                }, 2000 * Math.pow(2, reconnectAttempts));
            }
        };
    } catch (error) {
        console.error('Failed to create WebSocket:', error);
        updateConnectionStatus('error');
    }
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    try {
        const message = JSON.parse(data);
        
        switch (message.type) {
            case 'connection':
                currentSessionId = message.session_id;
                console.log('Session established:', currentSessionId);
                break;
                
            case 'processing':
                showTypingIndicator();
                break;
                
            case 'chunk':
                if (!currentResponse) {
                    hideWelcomeScreen();
                    currentResponse = addMessage('assistant', '');
                }
                if (currentResponse && message.content) {
                    currentResponse.textContent += message.content;
                    scrollToBottom();
                }
                break;
                
            case 'complete':
                hideTypingIndicator();
                if (message.message) {
                    if (currentResponse) {
                        currentResponse.textContent = message.message;
                    } else {
                        hideWelcomeScreen();
                        addMessage('assistant', message.message);
                    }
                }
                currentResponse = null;
                isGenerating = false;
                document.getElementById('sendBtn').disabled = document.getElementById('messageInput').value.trim() === '';
                
                // Save to conversation
                if (currentConversationId) {
                    saveMessageToConversation(currentConversationId, 'assistant', message.message || '');
                }
                break;
                
            case 'error':
                hideTypingIndicator();
                addMessage('system', `Error: ${message.message}`);
                currentResponse = null;
                isGenerating = false;
                document.getElementById('sendBtn').disabled = false;
                break;
                
            case 'pong':
                // Heartbeat response
                break;
        }
    } catch (e) {
        // Plain text message
        if (!currentResponse) {
            hideWelcomeScreen();
            currentResponse = addMessage('assistant', '');
        }
        if (currentResponse) {
            currentResponse.textContent += data;
            scrollToBottom();
        }
    }
}

// Send message
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || isGenerating) return;
    
    // Check WebSocket connection
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showNotification('Not connected. Reconnecting...', 'error');
        initWebSocket();
        return;
    }
    
    // Hide welcome screen if visible
    hideWelcomeScreen();
    
    // Display user message
    addMessage('user', message);
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    document.getElementById('sendBtn').disabled = true;
    isGenerating = true;
    
    // Create new conversation if needed
    if (!currentConversationId) {
        currentConversationId = await createConversation(message);
    }
    
    // Save user message to conversation
    if (currentConversationId) {
        await saveMessageToConversation(currentConversationId, 'user', message);
    }
    
    // Send via WebSocket
    const payload = {
        message: message,
        use_rag: useRAG,
        session_id: currentSessionId,
        conversation_id: currentConversationId
    };
    
    try {
        ws.send(JSON.stringify(payload));
        showTypingIndicator();
    } catch (error) {
        console.error('Error sending message:', error);
        showNotification('Failed to send message', 'error');
        isGenerating = false;
        document.getElementById('sendBtn').disabled = false;
    }
}

// Add message to chat
function addMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    // Create avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : role === 'assistant' ? 'J' : 'S';
    
    // Create content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    // Parse markdown if assistant message
    if (role === 'assistant' && content) {
        contentDiv.innerHTML = parseMarkdown(content);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
    
    return contentDiv;
}

// Parse basic markdown
function parseMarkdown(text) {
    // Code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

// Show/hide welcome screen
function hideWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatMessages = document.getElementById('chatMessages');
    
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    if (chatMessages) {
        chatMessages.style.display = 'block';
    }
}

function showWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatMessages = document.getElementById('chatMessages');
    
    if (welcomeScreen) {
        welcomeScreen.style.display = 'flex';
    }
    if (chatMessages) {
        chatMessages.style.display = 'none';
        chatMessages.innerHTML = '';
    }
}

// Typing indicator
function showTypingIndicator() {
    hideTypingIndicator(); // Remove any existing indicator
    
    const chatMessages = document.getElementById('chatMessages');
    const indicator = document.createElement('div');
    indicator.className = 'message assistant';
    indicator.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'J';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    indicator.appendChild(avatar);
    indicator.appendChild(content);
    chatMessages.appendChild(indicator);
    
    scrollToBottom();
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Conversation management
async function createConversation(firstMessage) {
    try {
        const response = await fetch('/api/conversations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: firstMessage.substring(0, 50) + (firstMessage.length > 50 ? '...' : ''),
                session_id: currentSessionId
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            await loadConversations(); // Refresh conversation list
            return data.id;
        }
    } catch (error) {
        console.error('Error creating conversation:', error);
    }
    return null;
}

async function loadConversations() {
    try {
        const response = await fetch('/api/conversations');
        if (response.ok) {
            const data = await response.json();
            conversations = data.conversations || [];
            displayConversations();
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

function displayConversations() {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    
    const todayConvs = [];
    const yesterdayConvs = [];
    const weekConvs = [];
    const olderConvs = [];
    
    conversations.forEach(conv => {
        const convDate = new Date(conv.created_at);
        const convItem = createConversationItem(conv);
        
        if (isSameDay(convDate, today)) {
            todayConvs.push(convItem);
        } else if (isSameDay(convDate, yesterday)) {
            yesterdayConvs.push(convItem);
        } else if (convDate > weekAgo) {
            weekConvs.push(convItem);
        } else {
            olderConvs.push(convItem);
        }
    });
    
    document.getElementById('todayConversations').innerHTML = todayConvs.join('');
    document.getElementById('yesterdayConversations').innerHTML = yesterdayConvs.join('');
    document.getElementById('weekConversations').innerHTML = weekConvs.join('');
    document.getElementById('olderConversations').innerHTML = olderConvs.join('');
}

function createConversationItem(conv) {
    const isActive = conv.id === currentConversationId ? 'active' : '';
    const date = new Date(conv.created_at);
    const dateStr = formatRelativeDate(date);
    
    return `
        <div class="conversation-item ${isActive}" onclick="loadConversation('${conv.id}')">
            <svg class="conversation-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
            </svg>
            <div class="conversation-text">
                <div class="conversation-title">${conv.title || 'New conversation'}</div>
                <div class="conversation-date">${dateStr}</div>
            </div>
            <div class="conversation-actions">
                <button class="action-btn" onclick="event.stopPropagation(); renameConversation('${conv.id}')" title="Rename">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path>
                    </svg>
                </button>
                <button class="action-btn" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="Delete">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                    </svg>
                </button>
            </div>
        </div>
    `;
}

async function loadConversation(conversationId) {
    try {
        const response = await fetch(`/api/conversations/${conversationId}`);
        if (response.ok) {
            const data = await response.json();
            currentConversationId = conversationId;
            
            // Update URL
            const url = new URL(window.location);
            url.searchParams.set('conversation', conversationId);
            window.history.pushState({}, '', url);
            
            // Clear chat and display messages
            hideWelcomeScreen();
            document.getElementById('chatMessages').innerHTML = '';
            
            // Display all messages
            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    addMessage(msg.role, msg.content);
                });
            } else {
                showWelcomeScreen();
            }
            
            // Update active conversation in sidebar
            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.closest('.conversation-item')?.classList.add('active');
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function saveMessageToConversation(conversationId, role, content) {
    try {
        await fetch(`/api/conversations/${conversationId}/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                role: role,
                content: content
            })
        });
    } catch (error) {
        console.error('Error saving message:', error);
    }
}

async function deleteConversation(conversationId) {
    if (!confirm('Delete this conversation?')) return;
    
    try {
        const response = await fetch(`/api/conversations/${conversationId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            if (conversationId === currentConversationId) {
                currentConversationId = null;
                showWelcomeScreen();
                
                // Clear URL
                const url = new URL(window.location);
                url.searchParams.delete('conversation');
                window.history.pushState({}, '', url);
            }
            await loadConversations();
        }
    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

async function renameConversation(conversationId) {
    const conv = conversations.find(c => c.id === conversationId);
    const newTitle = prompt('Enter new title:', conv?.title || '');
    
    if (newTitle && newTitle.trim()) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: newTitle.trim()
                })
            });
            
            if (response.ok) {
                // Update local conversations array
                const convIndex = conversations.findIndex(c => c.id === conversationId);
                if (convIndex !== -1) {
                    conversations[convIndex].title = newTitle.trim();
                }
                
                // Refresh the display
                await loadConversations();
                showNotification('Conversation renamed successfully', 'success');
            } else {
                showNotification('Failed to rename conversation', 'error');
            }
        } catch (error) {
            console.error('Error renaming conversation:', error);
            showNotification('Error renaming conversation', 'error');
        }
    }
}

function createNewChat() {
    currentConversationId = null;
    currentSessionId = null;
    showWelcomeScreen();
    
    // Clear URL
    const url = new URL(window.location);
    url.searchParams.delete('conversation');
    window.history.pushState({}, '', url);
    
    // Remove active state from all conversations
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
}

// UI Functions
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('collapsed');
}

function showSettings() {
    document.getElementById('settingsModal').classList.add('active');
}

function closeSettings() {
    document.getElementById('settingsModal').classList.remove('active');
}

function toggleRAG() {
    const toggle = document.getElementById('ragToggle');
    toggle.classList.toggle('active');
    useRAG = toggle.classList.contains('active');
    localStorage.setItem('useRAG', useRAG);
}

function loadSettings() {
    useRAG = localStorage.getItem('useRAG') !== 'false';
    const toggle = document.getElementById('ragToggle');
    if (toggle) {
        toggle.classList.toggle('active', useRAG);
    }
}

function showModelSelector() {
    // This would show a modal to select different models
    // For now, just navigate to models page
    window.location.href = '/models.html';
}

async function exportConversations() {
    try {
        const response = await fetch('/api/conversations/export');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `jarvis_conversations_${new Date().toISOString()}.json`;
            a.click();
            window.URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('Error exporting conversations:', error);
    }
}

async function clearAllConversations() {
    if (!confirm('This will delete all conversations. Are you sure?')) return;
    
    try {
        const response = await fetch('/api/conversations/clear', {
            method: 'DELETE'
        });
        
        if (response.ok) {
            conversations = [];
            currentConversationId = null;
            showWelcomeScreen();
            await loadConversations();
        }
    } catch (error) {
        console.error('Error clearing conversations:', error);
    }
}

// Utility Functions
function updateConnectionStatus(status) {
    const statusDot = document.getElementById('connectionStatus');
    const statusText = document.getElementById('connectionText');
    
    statusDot.className = 'status-dot';
    
    switch (status) {
        case 'connected':
            statusDot.classList.add('online');
            statusText.textContent = 'Connected';
            break;
        case 'connecting':
            statusDot.classList.add('connecting');
            statusText.textContent = 'Connecting...';
            break;
        case 'disconnected':
        case 'error':
            statusDot.classList.add('offline');
            statusText.textContent = 'Disconnected';
            break;
    }
}

function scrollToBottom() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function setupTextareaAutoResize() {
    const textarea = document.getElementById('messageInput');
    if (textarea) {
        textarea.addEventListener('input', () => autoResizeTextarea(textarea));
    }
}

function sendSuggestion(text) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = text;
    messageInput.focus();
    autoResizeTextarea(messageInput);
    document.getElementById('sendBtn').disabled = false;
}

function attachFile() {
    document.getElementById('fileInput').click();
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        // Handle file upload
        console.log('Files selected:', files);
        showNotification('File upload not yet implemented', 'info');
    }
}

function showNotification(message, type = 'info') {
    // Create a simple notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'error' ? 'var(--accent-red)' : 'var(--accent-blue)'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Date utilities
function isSameDay(date1, date2) {
    return date1.getFullYear() === date2.getFullYear() &&
           date1.getMonth() === date2.getMonth() &&
           date1.getDate() === date2.getDate();
}

function formatRelativeDate(date) {
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
        const hours = Math.floor(diff / (1000 * 60 * 60));
        if (hours === 0) {
            const minutes = Math.floor(diff / (1000 * 60));
            return minutes <= 1 ? 'Just now' : `${minutes} minutes ago`;
        }
        return hours === 1 ? '1 hour ago' : `${hours} hours ago`;
    } else if (days === 1) {
        return 'Yesterday';
    } else if (days < 7) {
        return `${days} days ago`;
    } else {
        return date.toLocaleDateString();
    }
}

// Heartbeat to keep connection alive
function startHeartbeat() {
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
