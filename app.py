from flask import Flask, render_template, Response, jsonify, send_from_directory, url_for
import cv2
from ultralytics import YOLO
import os
import numpy as np
import time

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configure Flask
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load YOLO model (downloads automatically if not present)
yolo = YOLO('yolov8n')

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Open the webcam
camera = cv2.VideoCapture(0)

# Make sure the history folder exists
if not os.path.exists('static/history'):
    os.makedirs('static/history')

# Store current alerts
current_alerts = []
camera_active = True
last_capture_time = 0
capture_interval = 5  # Minimum time between captures in seconds
violation_counter = 0  # Counter for violation filenames

# Helper function to save violation images
def save_violation(frame, violation_type):
    global last_capture_time, violation_counter
    current_time = time.time()
    
    # Only save if enough time has passed since last capture
    if current_time - last_capture_time >= capture_interval:
        violation_counter += 1
        filename = f"{violation_type}_{violation_counter}.jpg"
        filepath = os.path.join('static/history', filename)
        cv2.imwrite(filepath, frame)
        last_capture_time = current_time
        return filename
    return None

# Main detection function
def detect_violations(frame):
    alerts = []
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using both frontal and profile cascades
    faces_frontal = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces_profile = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Combine both detections
    faces = list(faces_frontal) + list(faces_profile)
    face_count = len(faces)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green color for face detection
    
    # Check for face violations
    if face_count == 0:
        alerts.append("No face detected!")
        save_violation(frame, 'no_face')
    elif face_count > 1:
        alerts.append("Multiple faces detected!")
        save_violation(frame, 'multiple_faces')
    elif len(faces_profile) > 0:
        alerts.append("Face turned away from screen!")
        save_violation(frame, 'face_turned')
    
    # Detect objects using YOLO
    results = yolo(frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            if class_name in ['cell phone', 'book', 'laptop']:
                alerts.append(f"{class_name.title()} detected!")
                save_violation(frame, class_name)
    
    # Update current alerts
    global current_alerts
    current_alerts = alerts
    return frame, alerts

# Video streaming generator
def generate_frames():
    while True:
        if camera_active:
            success, frame = camera.read()
            if not success:
                break
            frame, alerts = detect_violations(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send a black frame when camera is inactive
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_alerts')
def get_alerts():
    return jsonify(current_alerts)

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'status': 'success', 'camera_active': camera_active})

@app.route('/history')
def history():
    try:
        images = []
        history_dir = os.path.join(app.static_folder, 'history')
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        for filename in sorted(os.listdir(history_dir), reverse=True):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get violation type from filename
                violation_type = filename.split('_')[0].replace('-', ' ').title()
                
                images.append({
                    'filename': filename,
                    'violation_type': violation_type,
                    'url': url_for('static', filename=f'history/{filename}')
                })
        
        return render_template('history.html', images=images)
    except Exception as e:
        print(f"Error in history route: {str(e)}")
        return render_template('history.html', images=[], error=str(e))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)