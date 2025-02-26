let tts_audioContext;
let tts_socket;
let isMuted = true;

function init_tts_socket () {
    tts_socket = new WebSocket('ws://localhost:5052');
	tts_socket.binaryType = 'arraybuffer'

    tts_socket.onopen = () => {
        console.log('WebSocket TTS connection established');
    };

    tts_socket.onclose = () => {
        console.log('WebSocket TTS connection closed');
        // reconnect!
		init_tts_socket();
    };

    tts_socket.onmessage = (event) => {
        const audioData = event.data;
        //const pcmData = new Int16Array(audioData);

        if (!tts_audioContext || isMuted) {
            return;
        }

        // Decode and play audio
        tts_audioContext.decodeAudioData(audioData).then((audioBuffer) => {
            const source = tts_audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(tts_audioContext.destination);
            source.start(0);
        }).catch((error) => {
            console.error('Error decoding audio data:', error);
        });
    };

    tts_socket.onerror = (error) => {
        console.error('WebSocket TTS error:', error);
    };

}

document.addEventListener('DOMContentLoaded', (event) => {
    // Connect to the WebSocket server
    init_tts_socket();

    // Add event listener for the 'Unmute TTS' button
    document.getElementById('unmuteTTSBtn').addEventListener('click', () => {
        if (isMuted) {
            // Create an AudioContext
            const tts_AudioContext = window.AudioContext || window.webkitAudioContext;
            tts_audioContext = new tts_AudioContext();
            console.log('TTS AudioContext initialized');

            document.getElementById('unmuteTTSBtn').textContent = 'Mute TTS';
            isMuted = false;
        } else {
            // Mute the TTS
            if (tts_audioContext) {
                tts_audioContext.close();
                tts_audioContext = null;
                console.log('TTS AudioContext muted');
            }
            document.getElementById('unmuteTTSBtn').textContent = 'Unmute TTS';
            isMuted = true;
        }
    });
});

