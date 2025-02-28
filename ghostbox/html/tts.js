let TTS_SOURCE_WEBSOCK = 'ws://' + window.location.hostname + ':5052';
let tts_audioContext;
let tts_socket;
let isMuted = true;
let tts_audioBufferSource;

function init_tts_socket () {
	console.log("host: " + window.location.hostname);
    tts_socket = new WebSocket(TTS_SOURCE_WEBSOCK);
	tts_socket.binaryType = 'arraybuffer'

    tts_socket.onopen = () => {
        console.log('WebSocket TTS connection established');
    };

    tts_socket.onclose = () => {
        console.log('WebSocket TTS connection closed');
		try {
        tts_audioBufferSource.stop();
		} catch (err) {
			console.log("debug: caught " + err.name + " during regular closing of connection. Ignoring.");
			/* FIXME: damn you javascript, the error type is never in scope for instanceof
			if (err instanceof InvalidStateError) {
				console.log("debug: no biggie");
			} else {
				throw err;
			}*/
		}
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
            tts_audioContext = new AudioContext();
			tts_audioContext.resume()
            tts_audioBufferSource = tts_audioContext.createBufferSource();			
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

