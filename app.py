from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
# User 
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    role = db.Column(db.String(20), nullable=False)  # 'patient' / 'doctor' / 'admin'
    image_url = db.Column(db.String(255))

# patient
class Patient(User):
    __tablename__ = 'patient'
    id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    autism_survey_result = db.Column(db.String(50))
    adhd_result = db.Column(db.String(50))
    adhd_probabilities = db.Column(db.Text)
    face_result = db.Column(db.String(50))
    face_confidence = db.Column(db.String(50))

# doctor
class Doctor(User):
    __tablename__ = 'doctor'
    id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    is_active = db.Column(db.Boolean, default=True)

# Consultation
class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    diagnosis_id = db.Column(db.Integer, db.ForeignKey('diagnosis.id'))
    
    message_from_patient = db.Column(db.Text)
    message_from_doctor = db.Column(db.Text)
    status = db.Column(db.String(50), default='Pending')
    date_requested = db.Column(db.DateTime, default=datetime.utcnow)
    date_replied = db.Column(db.DateTime, nullable=True)

    diagnosis = db.relationship('Diagnosis')  # علشان نقدر نستدعي البيانات بسهولة
    patient = db.relationship('User', foreign_keys=[patient_id])


class Diagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    autism_survey_result = db.Column(db.String(50))
    adhd_result = db.Column(db.String(50))
    adhd_probabilities = db.Column(db.Text)
    face_result = db.Column(db.String(50))
    face_confidence = db.Column(db.String(50))
    input_data = db.Column(db.Text)

# Load Models
package = joblib.load('models/autism_survey_model_2v.pkl')
autism_model = package['model']
autism_scaler = package['scaler']

adhd_model = joblib.load('models/adhd_survey_model.pkl')
face_model = load_model('models/faceAutismModel1.h5')

# Helper: Encode Q-CHAT-10 Answers
def encode_answers(raw_answers):
    encoded = []
    for i in range(9):  # A1 to A9
        if raw_answers[i] in ['C', 'D', 'E']:
            encoded.append(1)
        else:
            encoded.append(0)
    if raw_answers[9] in ['A', 'B', 'C']:
        encoded.append(1)
    else:
        encoded.append(0)
    return encoded

# Routes

# Admin Account Initialization
def create_admin_account():
    existing_admin = User.query.filter_by(email='admin1@site.com').first()
    if not existing_admin:
        admin = User(
            full_name='Admin1',
            email='admin1@site.com',
            phone='0000000000',
            password=generate_password_hash('admin1@123'),
            gender='1',
            age=35,
            role='admin',
            image_url='static/uploads/admin1.jpg'
        )
        db.session.add(admin)
        db.session.commit()

@app.before_request
def before_any_request():
    if request.endpoint != 'static':
        create_admin_account()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        gender = request.form['gender']
        age = int(request.form['age'])
        photo = request.files['photo']

        if password != confirm_password:
            return "Passwords do not match."

        hashed_password = generate_password_hash(password)
        photo_path = f'static/uploads/{email}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        photo.save(photo_path)

        role = 'patient'
        new_user = User(
            full_name=full_name,
            email=email,
            phone=phone,
            password=hashed_password,
            gender=gender,
            age=age,
            role=role, 
            image_url=photo_path
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email_or_id = request.form['email_or_id']
        password = request.form['password']
        user = User.query.filter((User.email == email_or_id) | (User.id == email_or_id)).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.full_name
            session['user_email'] = user.email
            session['user_image'] = user.image_url
            session['user_gender'] = int(user.gender)
            session['user_age'] = int(user.age)
            session['role'] = user.role  # Store user role in session
            # Redirect based on user role
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('dashboard'))
        else:
            return "Invalid email/ID or password."
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/start-diagnosis')
def start_diagnosis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('step1'))

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        print("🔹 User Inputs from Step 1:")
        print(request.form.to_dict())  # Debugging line
        for i in range(1, 11):
            session[f"A{i}"] = request.form.get(f"A{i}")
        return redirect(url_for('step2'))
    return render_template('diagnosis_step1.html')

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        print("🔹 User Inputs from Step 1:")
        print(request.form.to_dict())  # Debugging line
        for i in range(1, 10):
            session[f"Q1_{i}"] = request.form.get(f"Q1_{i}")
        return redirect(url_for('step3'))
    return render_template('diagnosis_step2.html')

@app.route('/step3', methods=['GET', 'POST'])
def step3():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        print("🔹 User Inputs from Step 1:")
        print(request.form.to_dict())  # Debugging line
        for i in range(1, 10):
            session[f"Q2_{i}"] = request.form.get(f"Q2_{i}")
        return redirect(url_for('step4'))
    return render_template('diagnosis_step3.html')

@app.route('/step4', methods=['GET', 'POST'])
def step4():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        print("🔹 User Inputs from Step 1:")
        print(request.form.to_dict())  # Debugging line
        session['Daily_Phone_Usage_Hours'] = request.form.get('Daily_Phone_Usage_Hours')
        session['Difficulty_Organizing_Tasks'] = request.form.get('Difficulty_Organizing_Tasks')
        session['Focus_Score_Video'] = request.form.get('Focus_Score_Video')
        session['Daily_Coffee_Tea_Consumption'] = request.form.get('Daily_Coffee_Tea_Consumption')
        session['Learning_Difficulties'] = request.form.get('Learning_Difficulties')
        session['Anxiety_Depression_Levels'] = request.form.get('Anxiety_Depression_Levels')
        session['family_history'] = request.form.get('family_history')
        session['family_mem_with_ASD'] = request.form.get('family_mem_with_ASD')
        session['who_completed_the_test'] = request.form.get('who_completed_the_test')
        return redirect(url_for('diagnosis_result'))
    return render_template('diagnosis_step4.html')

@app.route('/diagnosis-result', methods=['GET'])
def diagnosis_result():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Retrieve autism questionnaire answers
    A_raw = [session.get(f"A{i}") for i in range(1, 11)]
    A_encoded = encode_answers(A_raw)

    autism_manual_score = sum(A_encoded)
    autism_manual_result = "Potential ASD Traits" if autism_manual_score > 3 else "No ASD Traits"

    # Get demographic and family history data
    age = int(session.get('user_age', 0))
    gender = int(session.get('user_gender', 1))
    family = int(session.get('family_mem_with_ASD', 0))
    who = int(session.get('who_completed_the_test', 0))

    # Prepare input for autism model
    autism_inputs = A_encoded + [age, gender, family, who]
    autism_scaled = autism_scaler.transform([autism_inputs])
    autism_output = autism_model.predict_proba(autism_scaled)[0]

    not_autistic_prob = autism_output[0] * 100
    autistic_prob = autism_output[1] * 100
    autism_survey_result = "Autistic" if autistic_prob > 50 else "Not Autistic"

    # Retrieve ADHD survey data
    Q1 = [int(session.get(f"Q1_{i}")) for i in range(1, 10)]
    Q2 = [int(session.get(f"Q2_{i}")) for i in range(1, 10)]

    # Retrieve additional inputs
    phone_usage = float(session.get('Daily_Phone_Usage_Hours'))
    organizing = int(session.get('Difficulty_Organizing_Tasks'))
    focus = int(session.get('Focus_Score_Video'))
    caffeine = int(session.get('Daily_Coffee_Tea_Consumption'))
    learning = int(session.get('Learning_Difficulties'))
    anxiety = int(session.get('Anxiety_Depression_Levels'))
    family_history = int(session.get('family_history'))

    # Predict ADHD result
    adhd_input = Q1 + Q2 + [phone_usage, organizing, focus, caffeine, learning, anxiety, family_history]
    adhd_probs = adhd_model.predict_proba([adhd_input])[0]
    adhd_labels = ["No ADHD", "Hyperactivity", "Inattention", "ADHD (Hyper + Inatt.)"]
    adhd_result = adhd_labels[np.argmax(adhd_probs)]
    adhd_prob_text = '\n'.join([f"{adhd_labels[i]}: {adhd_probs[i]*100:.2f}%" for i in range(4)])

    # Predict using the face image
    image_path = session.get('user_image', 'static/uploads/default.jpg')
    img = Image.open(image_path).resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    face_pred = face_model.predict(img)[0][0]
    face_result = 'Not Autistic' if face_pred > 0.5 else 'Autistic'
    face_confidence = face_pred * 100 if face_pred > 0.5 else (1 - face_pred) * 100

    # Save the full diagnosis record
    diagnosis = Diagnosis(
        user_id=session.get('user_id'),
        user_name=session.get('user_name'),
        email=session.get('user_email'),
        autism_survey_result=autism_survey_result,
        adhd_result=adhd_result,
        adhd_probabilities=adhd_prob_text,
        face_result=face_result,
        face_confidence=f"{face_confidence:.2f}%",
        input_data=str({
            'A_encoded': A_encoded,
            'Q1': Q1,
            'Q2': Q2,
            'secondary': {
                'phone_usage': phone_usage,
                'organizing': organizing,
                'focus_score': focus,
                'caffeine': caffeine,
                'learning': learning,
                'anxiety': anxiety,
                'family_history': family_history
            }
        })
    )
    db.session.add(diagnosis)
    db.session.commit()

    # Retrieve list of doctors to display in selection form
    doctors = User.query.filter_by(role='doctor').all()


    return render_template("diagnosis_result.html",
        autism_survey_result=autism_survey_result,
        autistic_prob=f"{autistic_prob:.2f}",
        not_autistic_prob=f"{not_autistic_prob:.2f}",
        autism_manual_result=autism_manual_result,
        autism_manual_score=autism_manual_score,
        adhd_result=adhd_result,
        adhd_probabilities=adhd_prob_text.replace('\n', '<br>'),
        face_result=face_result,
        face_confidence=f"{face_confidence:.2f}",
        doctors=doctors
    )

@app.route('/diagnosis-history')
def diagnosis_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get all diagnoses for this user
    records = Diagnosis.query.filter_by(user_id=session.get('user_id'))\
                             .order_by(Diagnosis.date.desc()).all()

    # Attach related consultation (if exists) for each diagnosis
    for r in records:
        r.consultation = Consultation.query.filter_by(diagnosis_id=r.id).first()

        # Attach doctor details if consultation exists
        if r.consultation:
            r.consultation.doctor = User.query.get(r.consultation.doctor_id)

    return render_template('diagnosis_history.html', records=records)

@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.full_name = request.form['full_name']
        user.phone = request.form['phone']
        user.age = int(request.form['age'])
        user.gender = request.form['gender']

        if 'photo' in request.files:
            photo = request.files['photo']
            if photo and photo.filename != '':
                photo_path = f'static/uploads/{user.email}.jpg'
                os.makedirs('static/uploads', exist_ok=True)
                photo.save(photo_path)
                user.image_url = photo_path
                session['user_image'] = photo_path  # Update session photo
        
        session['user_name'] = user.full_name  # Update name in session
        
        db.session.commit()
        return redirect(url_for('dashboard'))

    return render_template('edit_profile.html', user=user)


@app.route('/admin-dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        age = int(request.form['age'])
        gender = request.form['gender']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "This email is already registered."

        doctor = User(
            full_name=full_name,
            email=email,
            phone='N/A',
            password=generate_password_hash(password),
            gender=gender,
            age=age,
            role='doctor',
            image_url='static/uploads/default.jpg'
        )
        db.session.add(doctor)
        db.session.commit()

    doctors = User.query.filter_by(role='doctor').all()
    return render_template('admin_dashboard.html', doctors=doctors)


@app.route('/doctor-dashboard')
def doctor_dashboard():
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    doctor_id = session['user_id']

    # Retrieve consultations assigned to this doctor
    consultations = Consultation.query\
        .filter_by(doctor_id=doctor_id)\
        .order_by(Consultation.date_requested.desc()).all()

    return render_template('doctor_dashboard.html', consultations=consultations)


@app.route('/doctor-reply/<int:id>', methods=['POST'])
def doctor_reply(id):
    if 'user_id' not in session or session.get('role') != 'doctor':
        return redirect(url_for('login'))

    reply_text = request.form.get('reply')
    consultation = Consultation.query.get(id)

    if consultation and consultation.doctor_id == session['user_id']:
        consultation.message_from_doctor = reply_text
        consultation.status = 'Replied'
        consultation.date_replied = datetime.utcnow()
        db.session.commit()

    return redirect(url_for('doctor_dashboard'))


@app.route('/send-to-doctor', methods=['POST'])
def send_to_doctor():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor_id = int(request.form.get('doctor_id'))
    message = request.form.get('message')

    # Retrieve the latest diagnosis for the current user
    latest_diagnosis = Diagnosis.query.filter_by(user_id=session['user_id'])\
                                      .order_by(Diagnosis.date.desc()).first()

    if latest_diagnosis:
        consultation = Consultation(
            patient_id=session['user_id'],
            doctor_id=doctor_id,
            diagnosis_id=latest_diagnosis.id,
            message_from_patient=message
        )
        db.session.add(consultation)
        db.session.commit()

    return redirect(url_for('diagnosis_history'))

@app.route('/delete-doctor/<int:id>', methods=['POST'])
def delete_doctor(id):
    doctor = User.query.get_or_404(id)
    if doctor.role != 'doctor':
        return "Invalid operation."
    db.session.delete(doctor)
    db.session.commit()
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    