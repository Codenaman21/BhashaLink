from flask import Flask, render_template, request
from interpreter import process_text

app = Flask(__name__)

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
    text = ""
    video_files = []

   
    if request.method == 'POST':
        text = request.form.get('text', '')

    
    elif request.method == 'GET':
        text = request.args.get('text', '')

   
    if text.strip():
        video_files = process_text(text)

    return render_template('animation.html', video_sequence=video_files)

if __name__ == '__main__':
    app.run(debug=False)
