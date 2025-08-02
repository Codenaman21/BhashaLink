from flask import Flask, render_template, request, Response, jsonify
from interpreter import process_text
from sign_interpret import SignInterpreter
import cv2
import atexit

app = Flask(__name__)

interpreter = SignInterpreter()
cap = cv2.VideoCapture(0)

prediction = ""
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video-upload')
def video_upload():
    return render_template('video-upload.html')

@app.route('/interpret', methods=['GET', 'POST'])
def interpret():
    text = request.values.get('text', '').strip()
    video_files = []

    if text:
        video_files = process_text(text)

    return render_template('animation.html', video_sequence=video_files)

@app.route('/sign-live')
def sign_live():
    return render_template('temp.html', prediction=prediction, history=history)

@app.route('/predict_frame')
def predict_frame():
    def generate_frames():
        global cap, prediction, history
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            debug_frame, expression, _ = interpreter.process_frame(frame)

            if expression.strip() != "":
                prediction = expression.strip()
                if not history or (history and history[-1] != prediction):
                    history.append(prediction)

            ret, buffer = cv2.imencode('.jpg', debug_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(predict_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return {'expression': interpreter.expression.strip()}

@app.route('/start_camera')
def start_camera():
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return '', 204

@app.route('/stop_camera')
def stop_camera():
    global cap
    if cap.isOpened():
        cap.release()
    return '', 204

@app.route('/clear_prediction', methods=['POST'])
def clear_prediction():
    global prediction
    prediction = ""
    return jsonify({'status': 'success', 'message': 'Prediction cleared'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history.clear()
    return jsonify({'status': 'success', 'message': 'History cleared'})

def shutdown_camera():
    global cap
    if cap.isOpened():
        cap.release()

atexit.register(shutdown_camera)

if __name__ == '__main__':
    app.run(debug=False)
