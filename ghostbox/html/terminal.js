const TERMINAL_WEBSOCK_HOSTNAME = 'ws://localhost:5150';
let terminalSocket;


function ansiToHTML(text) {
    const ansiRegex = /\x1b\[[0-9;]*m/g;
    const htmlTags = {
        '0': '</span>', // Reset
        '1': '<span style="font-weight: bold;">', // Bold
        '3': '<span style="font-style: italic;">', // Italic
        '4': '<span style="text-decoration: underline;">', // Underline
        '30': '<span style="color: black;">', // Black
        '31': '<span style="color: red;">', // Red
        '32': '<span style="color: green;">', // Green
        '33': '<span style="color: yellow;">', // Yellow
        '34': '<span style="color: blue;">', // Blue
        '35': '<span style="color: magenta;">', // Magenta
        '36': '<span style="color: cyan;">', // Cyan
        '37': '<span style="color: white;">', // White
        '90': '<span style="color: grey;">', // Bright Black
        '91': '<span style="color: lightred;">', // Bright Red
        '92': '<span style="color: lightgreen;">', // Bright Green
        '93': '<span style="color: lightyellow;">', // Bright Yellow
        '94': '<span style="color: lightblue;">', // Bright Blue
        '95': '<span style="color: lightmagenta;">', // Bright Magenta
        '96': '<span style="color: lightcyan;">', // Bright Cyan
        '97': '<span style="color: lightwhite;">', // Bright White
        '40': '<span style="background-color: black;">', // Black background
        '41': '<span style="background-color: red;">', // Red background
        '42': '<span style="background-color: green;">', // Green background
        '43': '<span style="background-color: yellow;">', // Yellow background
        '44': '<span style="background-color: blue;">', // Blue background
        '45': '<span style="background-color: magenta;">', // Magenta background
        '46': '<span style="background-color: cyan;">', // Cyan background
        '47': '<span style="background-color: white;">', // White background
        '100': '<span style="background-color: grey;">', // Bright Black background
        '101': '<span style="background-color: lightred;">', // Bright Red background
        '102': '<span style="background-color: lightgreen;">', // Bright Green background
        '103': '<span style="background-color: lightyellow;">', // Bright Yellow background
        '104': '<span style="background-color: lightblue;">', // Bright Blue background
        '105': '<span style="background-color: lightmagenta;">', // Bright Magenta background
        '106': '<span style="background-color: lightcyan;">', // Bright Cyan background
        '107': '<span style="background-color: lightwhite;">' // Bright White background
    };

    let result = '';
    let lastIndex = 0;

    text.replace(ansiRegex, (match, offset) => {
        const code = match.slice(2, -1);
        const codes = code.split(';').map(Number);

        if (codes.includes(0)) {
            result += text.slice(lastIndex, offset) + '</span>';
        } else {
            result += text.slice(lastIndex, offset);
            codes.forEach(code => {
                if (htmlTags[code]) {
                    result += htmlTags[code];
                }
            });
        }

        lastIndex = offset + match.length;
    });

    result += text.slice(lastIndex);

    return result;
}

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
        const stderrToken = "[|STDER|]:";
        const l = stderrToken.length;
        if (message.startsWith(stderrToken)) {
            addMessageToConsoleLog(message.substring(l));
        } else {
            addMessageToPage(message);
        }
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
    // and replace ansi codes with corresponding html tags
    const formattedMessage = ansiToHTML(message.replace(/\n/g, '<br>'));

    // Set the inner HTML to handle the <br> tags
    messageElement.innerHTML = formattedMessage;
    messagesDiv.appendChild(messageElement);
    messageElement.classList.add('show');
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the bottom
}

function addMessageToConsoleLog(message) {
    const consoleDiv = document.getElementById('consoleLog');
    const messageElement = document.createElement('span');
    messageElement.className = 'console-message';

    // Replace newlines with <br> tags
    // and replace ansi codes with corresponding html tags
    const formattedMessage = ansiToHTML(message.replace(/\n/g, '<br>'));

    // Set the inner HTML to handle the <br> tags
    messageElement.innerHTML = formattedMessage;
    consoleDiv.appendChild(messageElement);
    messageElement.classList.add('show');
    consoleDiv.scrollTop = consoleDiv.scrollHeight; // Scroll to the bottom
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

    function sendCommand(command) {
        if (command.trim() !== '') {
            addUserMessageToPage(command);
            terminalSocket.send(command);                
        }
    }
        
    document.getElementById('submit_input_button').addEventListener('click', () => {
        const command = commandInput.value;
        sendCommand(command);
        commandInput.value = '';                        
    });
                                                                    
    commandInput.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            const command = commandInput.value;
            sendCommand(command);
            commandInput.value = '';                
        }
    });
});
