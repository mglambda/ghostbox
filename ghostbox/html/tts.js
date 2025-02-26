let tts_audioContext;
let tts_socket;
let isMuted = true;
let tts_audioBuffer;

function init_tts_socket () {
    tts_socket = new WebSocket('ws://localhost:5052');
	tts_socket.binaryType = 'arraybuffer'

    tts_socket.onopen = () => {
        console.log('WebSocket TTS connection established');
    };

    tts_socket.onclose = () => {
        console.log('WebSocket TTS connection closed');
        tts_audioBufferSource.stop()
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
            tts_audioBufferSource = tts_audioContext.createBufferSource();
            tts_audioBufferSource.buffer = audioBuffer;
			tts_audioBufferSource.onended = () => {
				tts_socket.send("done");
			};
			
            tts_audioBufferSource.connect(tts_audioContext.destination);
            tts_audioBufferSource.start(0);
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

