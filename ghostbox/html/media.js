let MIC_DESTINATION_WEBSOCK = 'ws://' + window.location.hostname + ':5051';
let audioContext;
let mediaStreamAudioSourceNode;
let scriptProcessorNode;
let media_socket;

document.getElementById('startBtn').addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Attempt to create an AudioContext with a sample rate of 44100 Hz

    const AudioContext = window.AudioContext || window.webkitAudioContext;
		// update: nope, browsers don't like this
    //const options = { sampleRate: 44100 };
const options = {};	
    audioContext = new AudioContext(options);
    console.log('AudioContext sample rate:', audioContext.sampleRate);
	
    mediaStreamAudioSourceNode = audioContext.createMediaStreamSource(stream);
    scriptProcessorNode = audioContext.createScriptProcessor(1024, 1, 1);

    // Connect the nodes
    mediaStreamAudioSourceNode.connect(scriptProcessorNode);
    scriptProcessorNode.connect(audioContext.destination);

    // Connect to the WebSocket server
    media_socket = new WebSocket(MIC_DESTINATION_WEBSOCK);

    media_socket.onopen = () => {
        console.log('WebSocket connection established');
		media_socket.send("samplerate:" + audioContext.sampleRate);
    };

    media_socket.onclose = () => {
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

        if (media_socket.readyState === WebSocket.OPEN) {
            media_socket.send(pcmData.buffer);
        }
    };

    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
});

document.getElementById('stopBtn').addEventListener('click', () => {
    if (media_socket.readyState === WebSocket.OPEN) {
        media_socket.send('END');
        media_socket.close();
    }
    mediaStreamAudioSourceNode.disconnect();
    scriptProcessorNode.disconnect();
    audioContext.close();
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
});
