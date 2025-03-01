const TERMINAL_WEBSOCK_HOSTNAME = 'ws://localhost:5150';
let terminalSocket;

function initTerminalSocket() {
    terminalSocket = new WebSocket(TERMINAL_WEBSOCK_HOSTNAME);

    terminalSocket.onopen = () => {
        console.log('WebSocket terminal connection established');
    };

    terminalSocket.onclose = () => {
        console.log('WebSocket terminal connection closed');
        // Reconnect!
        initTerminalSocket();
    };

    terminalSocket.onmessage = (event) => {
        const message = event.data;
        addMessageToPage(message);
    };

    terminalSocket.onerror = (error) => {
        console.error('WebSocket terminal error:', error);
    };
}

function addMessageToPage(message) {
    const messagesDiv = document.getElementById('messages');
    const messageElement = document.createElement('span');
    messageElement.className = 'message';

    // Replace newlines with <br> tags
    const formattedMessage = message.replace(/\n/g, '<br>');

    // Set the inner HTML to handle the <br> tags
    messageElement.innerHTML = formattedMessage;
    messagesDiv.appendChild(messageElement);
    messageElement.classList.add('show');
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the bottom
}


function addUserMessageToPage(message) {
    const messagesDiv = document.getElementById('messages');
    const messageElement = document.createElement('span');
    messageElement.className = 'user-message';

    // Replace newlines with <br> tags
    const formattedMessage = message.replace(/\n/g, '<br>') ;

    // we know that the user message at least comes in one chunk, so we can wrap it in <br>s
    messageElement.innerHTML = "<br>" + formattedMessage + "<br>";
    messagesDiv.appendChild(messageElement);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the bottom
}

document.addEventListener('DOMContentLoaded', (event) => {
    // Connect to the WebSocket server
    initTerminalSocket();

    const commandInput = document.getElementById('commandInput');

    commandInput.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            const command = commandInput.value;
            if (command.trim() !== '') {
                addUserMessageToPage(command);
                terminalSocket.send(command);
                commandInput.value = '';
            }
        }
    });
});
