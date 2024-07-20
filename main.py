from flask import Flask, render_template, request, redirect, url_for, flash
import os
import urllib.request
from werkzeug.utils import secure_filename
from gesture_recognizer import recognize_gestures_and_display

app = Flask(__name__)
app.secret_key = "supersecretkey"  
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

    #download images if they do not exist
    image_paths = []
    for name in IMAGE_FILENAMES:
        image_path = os.path.join('static', 'images', name)
        image_paths.append(image_path)
        if not os.path.exists(image_path):
            url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
            urllib.request.urlretrieve(url, image_path)

    # call the gesture recognition function and save the output to a file
    output_path = os.path.join('static', 'images', 'gesture_visualization.png')
    recognize_gestures_and_display(image_paths, output_path)

    return render_template('visualize.html', image_url=url_for('static', filename='images/gesture_visualization.png'))
###
@app.route('/index', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file')
        image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_paths.append(file_path)
            else:
                flash('Allowed file types are png, jpg, jpeg')
                return redirect(request.url)

        if image_paths:
            output_path = os.path.join('static', 'images', 'user_gesture_visualization.png')
            recognize_gestures_and_display(image_paths, output_path)
            return render_template('index.html', image_url=url_for('static', filename='images/user_gesture_visualization.png'))
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
