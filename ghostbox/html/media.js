let audioContext;
let mediaStreamAudioSourceNode;
let scriptProcessorNode;
let socket;

document.getElementById('startBtn').addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Attempt to create an AudioContext with a sample rate of 44100 Hz
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const options = { sampleRate: 44100 };
//const options = {};	
    audioContext = new AudioContext(options);
    console.log('AudioContext sample rate:', audioContext.sampleRate);
	
    mediaStreamAudioSourceNode = audioContext.createMediaStreamSource(stream);
    scriptProcessorNode = audioContext.createScriptProcessor(1024, 1, 1);

    // Connect the nodes
    mediaStreamAudioSourceNode.connect(scriptProcessorNode);
    scriptProcessorNode.connect(audioContext.destination);

    // Connect to the WebSocket server
    socket = new WebSocket('ws://localhost:5051');

    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed');
    };

    scriptProcessorNode.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
		//        const outputData = new Float32Array(inputData.length);
        const outputData = audioContext.createBuffer(1, inputData.length, audioContext.sampleRate);

        for (let i = 0; i < inputData.length; i++) {
            outputData[i] = inputData[i];
        }

        const pcmData = new Int16Array(outputData.length);
        for (let i = 0; i < outputData.length; i++) {
            pcmData[i] = Math.min(1, Math.max(-1, outputData[i])) * 0x7FFF;
        }

        if (socket.readyState === WebSocket.OPEN) {
            socket.send(pcmData.buffer);
        }
    };

    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
});

document.getElementById('stopBtn').addEventListener('click', () => {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send('END');
        socket.close();
    }
    mediaStreamAudioSourceNode.disconnect();
    scriptProcessorNode.disconnect();
    audioContext.close();
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
});
