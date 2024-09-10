import whisper

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

model = whisper.load_model("base.en")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Check if the 'file' part is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        # Check if the user has selected a file
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)

        # Load the audio file and perform transcription
        audio = whisper.load_audio(temp_file_path)
        result = model.transcribe(audio)

        # Clean up the temporary file and directory
        os.remove(temp_file_path)
        os.rmdir(temp_dir)

        return jsonify({'text': result['text']}), 200
    
    except whisper.WhisperError as e:
        return jsonify({'error': 'Whisper model error', 'message': str(e)}), 500
    
    except Exception as e:
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
