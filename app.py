from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit
import os
import joblib
import cv2

app = Flask(__name__)

# Set up your file and database configurations
app.secret_key = 'mo9652150077'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
socketio = SocketIO(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True) 
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(50), nullable=True)

# Create the database tables inside the application context
with app.app_context():
    db.create_all()

# Helper function for allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the model and label encoder
def load_model():
    model = joblib.load('emotion_model.pkl')  # Load the trained model
    label_encoder = joblib.load('label_encoder.pkl')  # Load the trained label encoder
    return model, label_encoder
# Root Route
@app.route('/')
def home():
    return redirect(url_for('login'))  # Redirects to the login page

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(username=username, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('User already exists.', 'danger')
    return render_template('register.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# Reset Password Route
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        flash('Password reset feature not implemented.', 'info')
    return render_template('reset_password.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# Profile Route (New)
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        flash('Please log in to view your profile.', 'warning')
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['user']).first()
    
    if request.method == 'POST':
        user.email = request.form['email']
        user.age = request.form['age']
        user.gender = request.form['gender']
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    
    return render_template('profile.html', user=user)

# FAQ Route (New)
@app.route('/faq')
def faq():
    return render_template('faq.html')

# Contact Support Route (New)
@app.route('/contact-support', methods=['GET', 'POST'])
def contact_support():
    if request.method == 'POST':
        subject = request.form['subject']
        message = request.form['message']
        flash(f'Support message sent! Subject: {subject} Message: {message}', 'success')
    return render_template('contact_support.html')

# Search History Route (New)
@app.route('/search-history', methods=['GET', 'POST'])
def search_history():
    if 'user' not in session:
        flash('Please log in to search analysis history.', 'warning')
        return redirect(url_for('login'))

    results = []
    if request.method == 'POST':
        emotion = request.form['emotion']
        # In a real app, you would query the database or search logs
        results = [
            {'emotion': emotion, 'date': '2024-12-01'},
            {'emotion': emotion, 'date': '2024-12-02'}
        ]
    return render_template('search_history.html', results=results)

# Generate Report Route (New)
@app.route('/generate-report', methods=['GET', 'POST'])
def generate_report():
    if request.method == 'POST':
        emotion = request.form['emotion']
        # Logic for generating the report (e.g., calling the report generation function)
        report_content = f"Generated Report for Emotion: {emotion}\n"
        report_content += "Additional details can be added here.\n"
        
        flash('Report generated successfully!', 'success')
        
        # Pass the report content to the template
        return render_template('view_report.html', report=report_content)
    return render_template('generate_report.html')

# Image Upload Route
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if 'user' not in session:
        flash('Please log in to upload images.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Emit progress update for upload
            socketio.emit('progress_update', {'progress': 50})

            # Perform face detection and emotion prediction
            emotion = face_detect_and_predict(filepath)

            # Emit progress update for completion
            socketio.emit('progress_update', {'progress': 100})

            # Flash message and remain on the same page
            flash(f'File uploaded and processed. Emotion detected: {emotion}', 'success')

            # Re-render the upload page, showing the flash message
            return render_template('upload.html')

        else:
            flash('Invalid file type.', 'danger')
            return render_template('upload.html')

    return render_template('upload.html')

# Function to detect face and predict emotion
def face_detect_and_predict(image_path):
    model, label_encoder = load_model()

    # Load the uploaded image
    image = cv2.imread(image_path)

    # Resize the image to 224x224 for model compatibility
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_resized = cv2.resize(image, (224, 224))  # Resize to 224x224

    # Flatten the image to match the model's expected input shape
    image_flattened = image_resized.flatten().reshape(1, -1)  # Flatten and reshape to (1, 150528)

    # Predict the emotion
    emotion_prediction = model.predict(image_flattened)
    emotion = label_encoder.inverse_transform(emotion_prediction)

    flash(f'Face detected! Predicted emotion: {emotion[0]}', 'success')
    return emotion[0]  # Return the predicted emotion

@socketio.on('progress_request')
def handle_progress_request():
    emit('progress_update', {'progress': 0})

# Main Execution
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    socketio.run(app, debug=True)
