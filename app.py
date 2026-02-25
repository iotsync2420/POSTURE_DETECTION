import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template_string
from main import PostureAnalyzer

app = Flask(__name__)
analyzer = PostureAnalyzer()

# A simple HTML interface so you can see it working
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Posture Detector</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #1a1a1a; color: white; }
        video, canvas { width: 100%; max-width: 500px; border-radius: 10px; }
        .alert { color: #ff4444; font-weight: bold; font-size: 1.2em; }
        .good { color: #00ff00; }
    </style>
</head>
<body>
    <h1>Posture Detection</h1>
    <video id="video" autoplay></video>
    <div id="result">Initializing...</div>
    <canvas id="canvas" style="display:none;"></canvas>

    <script>
        const video = document.getElementById('video');
        const resultDiv = document.getElementById('result');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; });

        setInterval(() => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const data = canvas.toDataURL('image/jpeg');

            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: data })
            })
            .then(res => res.json())
            .then(data => {
                if(data.status === 'success') {
                    let html = `<p>Position: ${data.position}</p>`;
                    data.issues.forEach(i => {
                        html += `<p class="${i.type === 'good_posture' ? 'good' : 'alert'}">${i.text}</p>`;
                    });
                    resultDiv.innerHTML = html;
                }
            });
        }, 500); // Sends a frame every 0.5 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json['image']
        _, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = analyzer.run_on_frame(frame)
        
        if results["status"] == "success":
            # Map the results to a cleaner format for the frontend
            issue_list = [{"type": i[0], "text": i[1]} for i in results["issues"]]
            return jsonify({"status": "success", "position": results["position"], "issues": issue_list})
        
        return jsonify({"status": "no_person"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)