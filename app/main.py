from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_from_directory
from markupsafe import Markup
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import uuid
import json
import tensorflow as tf
import os
import requests
import re
from flask_login import UserMixin, login_user, logout_user, LoginManager, login_required, current_user
import pandas as pd
import sklearn
import pickle
import firebase_admin
from firebase_admin import credentials, firestore, auth
from deep_translator import GoogleTranslator
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables from .env file
# Create a .env file in your project root with the following keys:
#   SECRET_KEY=your_flask_secret
#   TOGETHER_API_KEY=your_together_api_key
#   FIREBASE_KEY_PATH=secrets/firebase_key.json   (or absolute path)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'change-me-in-production')

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration  (single source of truth — no duplicate definitions)
# ─────────────────────────────────────────────────────────────────────────────
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')
TOGETHER_API_URL = 'https://api.together.xyz/v1/chat/completions'

# Model used for plant disease AI fallback and chatbot
AI_MODEL_CHAT     = 'mistralai/Mistral-7B-Instruct-v0.1'
AI_MODEL_DISEASE  = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'

# ─────────────────────────────────────────────────────────────────────────────
# Firebase Initialization
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
cred_path = os.getenv('FIREBASE_KEY_PATH',
                      os.path.join(BASE_DIR, 'secrets', 'firebase_key.json'))

db            = None
firebase_auth = None   # FIX: single, unambiguous reference — never overwritten

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    db            = firestore.client()
    firebase_auth = auth          # FIX: assigned once, never reassigned below
    print('Firebase connected successfully!')
except Exception as e:
    print('Firebase init error:', e)

# ─────────────────────────────────────────────────────────────────────────────
# ML Model Loading  (clear, distinct names for each model)
# ─────────────────────────────────────────────────────────────────────────────
# Crop recommendation models
try:
    crop_model  = pickle.load(open('model.pkl',          'rb'))
    crop_sc     = pickle.load(open('standscaler.pkl',    'rb'))
    crop_mx     = pickle.load(open('minmaxscaler.pkl',   'rb'))
    print('Crop ML models loaded successfully!')
except Exception as e:
    print(f'Error loading crop ML models: {e}')
    crop_model, crop_sc, crop_mx = None, None, None

# Plant disease detection model
DISEASE_MODEL_PATH = 'models/plant_disease_recog_model_pwp.keras'
try:
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    print('Plant disease model loaded successfully!')
except Exception as e:
    print(f'Error loading disease model: {e}')
    disease_model = None

# ─────────────────────────────────────────────────────────────────────────────
# File Upload Configuration  (single definition — no duplicates)
# ─────────────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER_PRODUCTS = os.path.join(BASE_DIR, 'static', 'images')
UPLOAD_FOLDER_DISEASE  = os.path.join(BASE_DIR, 'uploadimages')
ALLOWED_EXTENSIONS     = {'png', 'jpg', 'jpeg', 'gif'}

for folder in (UPLOAD_FOLDER_PRODUCTS, UPLOAD_FOLDER_DISEASE):
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']        = UPLOAD_FOLDER_PRODUCTS
app.config['UPLOAD_FOLDER_DISEASE'] = UPLOAD_FOLDER_DISEASE
app.config['MAX_CONTENT_LENGTH']   = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    """FIX: single definition used everywhere."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────────────────────────────────────
# Plant Disease Labels & Info
# ─────────────────────────────────────────────────────────────────────────────
label = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

with open('plant_disease.json', 'r') as file:
    plant_disease_info = json.load(file)

CONFIDENCE_THRESHOLD = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# Flask-Login Setup
# ─────────────────────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, uid):
        self.id = uid

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ─────────────────────────────────────────────────────────────────────────────
# Translation Helper  (single helper — use everywhere instead of inline lookup)
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATIONS = {
    'en': {
        'home': 'Home', 'agro_products': 'Agro Products', 'ai_help': 'AI Help',
        'gov_schems': 'Government Schemes', 'hire_labour': 'Hire Labour',
        'soil_test': 'Soil Test', 'equipment_rental': 'Equipment Rental',
        'more': 'More', 'smart_agriculture': 'Smart Agriculture',
        'farmer_register': 'Farmer Register', 'add_farming': 'Add Farming',
        'farmer_details': 'Farmer Details', 'smart_farm': 'Smart Farm',
        'map': 'Map', 'live_prices': 'Live Prices', 'transport': 'Transport',
        'disease': 'Disease', 'plant_condition': 'Plant Condition', 'cart': 'Cart',
        'crop_roadmap': 'Crop Road Map', 'welcome': 'Welcome', 'logout': 'Logout',
        'signup': 'Signup', 'main_title': 'Smart Agriculture',
        'intro_subtitle': 'Revolutionizing Farming with Technology',
        'agro_products_button': 'AGRO PRODUCTS',
        'plant_disease_detection': 'Plant Disease Detection',
        'fertilizer_prediction': 'Fertiliser Prediction', 'farmer_name': 'Farmer Name',
        'aadhaar_number': 'Aadhaar Number', 'age': 'Age', 'gender': 'Gender',
        'male': 'Male', 'phone_number': 'Phone Number', 'address': 'Address',
        'farming_type': 'Farming Type', 'farming': 'FARMING',
        'register_farmer': 'Register Farmer', 'farmer_registration': 'Farmer Registration',
        'add': 'ADD', 'delete_confirmation': 'Are you sure to Delete data',
        'rid': 'RID', 'delete': 'DELETE', 'add_agro_product': 'ADD AGRO PRODUCT',
        'add_agro_products_title': 'Add AgroProducts',
        'add_agro_products_heading': 'Add Agro Products', 'farmer_email': 'Farmer Email',
        'farmer_email_placeholder': 'yourname@gmail.com', 'product_name': 'Product Name',
        'upload_product_image': 'Upload Product Image', 'choose_file': 'Choose File',
        'no_file_chosen': 'No file chosen',
        'image_upload_optional': 'Select an image file (JPG, PNG, etc.). Optional.',
        'product_description': 'Product Description', 'price': 'Price', 'unit': 'Unit',
        'kg': 'kg', 'quintal': 'quintal', 'add_product_button': 'Add Product',
        'farmer_services': 'Farmer Services', 'resources': 'Resources',
        'ai_tools': 'AI Tools', 'invalid_farming_type': 'Invalid farming type selected.',
        'registration_error': 'Registration error: {e}',
    },
    'hi': {
        'home': 'होम', 'agro_products': 'कृषि उत्पाद', 'gov_schems': 'सरकारी योजनाएं',
        'ai_help': 'एआई मदद', 'hire_labour': 'श्रमिक किराए पर लें',
        'soil_test': 'मिट्टी परीक्षण', 'equipment_rental': 'उपकरण किराये',
        'more': 'और अधिक', 'smart_agriculture': 'स्मार्ट कृषि',
        'farmer_register': 'किसान पंजीकरण', 'add_farming': 'खेती जोड़ें',
        'farmer_details': 'किसान विवरण', 'smart_farm': 'स्मार्ट खेत',
        'map': 'नक्शा', 'live_prices': 'लाइव कीमतें', 'transport': 'परिवहन',
        'disease': 'रोग', 'plant_condition': 'पौधे की स्थिति', 'cart': 'कार्ट',
        'crop_roadmap': 'फसल रोड मैप', 'welcome': 'स्वागत है', 'logout': 'लॉग आउट',
        'signup': 'साइन अप करें', 'main_title': 'स्मार्ट कृषि',
        'intro_subtitle': 'प्रौद्योगिकी से खेती में क्रांति',
        'agro_products_button': 'कृषि उत्पाद',
        'plant_disease_detection': 'पौधे की बीमारी का पता लगाना',
        'fertilizer_prediction': 'उर्वरक पूर्वानुमान', 'farmer_name': 'किसान का नाम',
        'aadhaar_number': 'आधार नंबर', 'age': 'आयु', 'gender': 'लिंग', 'male': 'पुरुष',
        'phone_number': 'फ़ोन नंबर', 'address': 'पता', 'farming_type': 'खेती का प्रकार',
        'register_farmer': 'किसान को पंजीकृत करें',
        'farmer_registration': 'किसान पंजीकरण', 'add': 'जोड़ें',
        'delete_confirmation': 'क्या आप डेटा हटाना चाहते हैं?', 'delete': 'हटाएं',
        'rid': 'आरआईडी', 'farming': 'खेती',
        'add_agro_products_title': 'कृषि उत्पाद जोड़ें',
        'add_agro_products_heading': 'कृषि उत्पाद जोड़ें', 'farmer_email': 'किसान ईमेल',
        'farmer_email_placeholder': 'yourname@gmail.com', 'product_name': 'उत्पाद का नाम',
        'upload_product_image': 'उत्पाद की छवि अपलोड करें', 'choose_file': 'फ़ाइल चुनें',
        'no_file_chosen': 'कोई फ़ाइल नहीं चुनी गई',
        'image_upload_optional': 'एक छवि फ़ाइल चुनें (JPG, PNG, आदि)। वैकल्पिक।',
        'product_description': 'उत्पाद विवरण', 'price': 'मूल्य', 'unit': 'इकाई',
        'kg': 'किग्रा', 'quintal': 'क्विंटल', 'add_product_button': 'उत्पाद जोड़ें',
        'farmer_services': 'किसान सेवाएं', 'resources': 'संसाधन', 'ai_tools': 'एआई उपकरण',
        'invalid_farming_type': 'अमान्य खेती प्रकार चुना गया।',
        'registration_error': 'पंजीकरण त्रुटि: {e}',
    },
    'kn': {
        'home': 'ಹೋಮ್', 'agro_products': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳು',
        'gov_schems': 'ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು', 'ai_help': 'ಎಐ ಸಹಾಯ',
        'hire_labour': 'ಕಾರ್ಮಿಕರನ್ನು ನೇಮಿಸಿಕೊಳ್ಳಿ', 'soil_test': 'ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ',
        'equipment_rental': 'ಉಪಕರಣ ಬಾಡಿಗೆ', 'more': 'ಇನ್ನಷ್ಟು',
        'smart_agriculture': 'ಸ್ಮಾರ್ಟ್ ಕೃಷಿ', 'farmer_register': 'ರೈತ ನೋಂದಣಿ',
        'add_farming': 'ಕೃಷಿ ಸೇರಿಸಿ', 'farmer_details': 'ರೈತರ ವಿವರಗಳು',
        'smart_farm': 'ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮ್', 'map': 'ನಕ್ಷೆ',
        'live_prices': 'ಲೈವ್ ಬೆಲೆಗಳು', 'transport': 'ಸಾರಿಗೆ', 'disease': 'ರೋಗ',
        'plant_condition': 'ಸಸ್ಯ ಸ್ಥಿತಿ', 'cart': 'ಕಾರ್ಟ್',
        'crop_roadmap': 'ಬೆಳೆ ಮಾರ್ಗ ನಕ್ಷೆ', 'welcome': 'ಸ್ವಾಗತ',
        'logout': 'ಲಾಗ್ ಔಟ್', 'signup': 'ಸೈನ್ ಅಪ್', 'main_title': 'ಸ್ಮಾರ್ಟ್ ಕೃಷಿ',
        'intro_subtitle': 'ತಂತ್ರಜ್ಞಾನದಿಂದ ಕೃಷಿಯಲ್ಲಿ ಕ್ರಾಂತಿ',
        'agro_products_button': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳು',
        'plant_disease_detection': 'ಮೂಲಕ ಬೆಳೆ ರೋಗ ಪತ್ತೆ',
        'fertilizer_prediction': 'ರಸಗೊಬ್ಬರ ಮುನ್ಸೂಚನೆ', 'rid': 'RID',
        'farmer_name': 'ರೈತರ ಹೆಸರು', 'aadhaar_number': 'ಆಧಾರ್ ಸಂಖ್ಯೆ', 'age': 'ವಯಸ್ಸು',
        'gender': 'ಲಿಂಗ', 'male': 'ಪುರುಷ', 'phone_number': 'ದೂರವಾಣಿ ಸಂಖ್ಯೆ',
        'address': 'ವಿಳಾಸ', 'farming_type': 'ಕೃಷಿ ಪ್ರಕಾರ',
        'register_farmer': 'ರೈತರನ್ನು ನೋಂದಾಯಿಸಿ', 'farmer_registration': 'ರೈತ ನೋಂದಣಿ',
        'add': 'ಸೇರಿಸಿ', 'delete_confirmation': 'ಡೇಟಾವನ್ನು ಅಳಿಸಲು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ?',
        'farming': 'ಕೃಷಿ', 'delete': 'ಅಳಿಸಿ',
        'add_agro_product': 'ಕೃಷಿ ಉತ್ಪನ್ನ ಸೇರಿಸಿ',
        'add_agro_products_title': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳನ್ನು ಸೇರಿಸಿ',
        'add_agro_products_heading': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳನ್ನು ಸೇರಿಸಿ',
        'farmer_email': 'ರೈತರ ಇಮೇಲ್',
        'farmer_email_placeholder': 'farmername@gmail.com', 'product_name': 'ಉತ್ಪನ್ನದ ಹೆಸರು',
        'upload_product_image': 'ಉತ್ಪನ್ನದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ',
        'choose_file': 'ಫೈಲ್ ಆಯ್ಕೆಮಾಡಿ', 'no_file_chosen': 'ಯಾವುದೇ ಫೈಲ್ ಆಯ್ಕೆ ಮಾಡಲಾಗಿಲ್ಲ',
        'image_upload_optional': 'ಚಿತ್ರ ಫೈಲ್ ಆಯ್ಕೆಮಾಡಿ (JPG, PNG, ಇತ್ಯಾದಿ). ಐಚ್ಛಿಕ.',
        'product_description': 'ಉತ್ಪನ್ನ ವಿವರಣೆ', 'price': 'ಬೆಲೆ', 'unit': 'ಘಟಕ',
        'kg': 'ಕೆಜಿ', 'quintal': 'ಕ್ವಿಂಟಲ್', 'add_product_button': 'ಉತ್ಪನ್ನ ಸೇರಿಸಿ',
        'farmer_services': 'ರೈತ ಸೇವೆಗಳು', 'resources': 'ಸಂಪನ್ಮೂಲಗಳು',
        'ai_tools': 'AI ಉಪಕರಣಗಳು',
        'invalid_farming_type': 'ಅಮಾನ್ಯ ಕೃಷಿ ಪ್ರಕಾರ ಆಯ್ಕೆ ಮಾಡಲಾಗಿದೆ.',
        'registration_error': 'ನೋಂದಣಿ ದೋಷ: {e}',
    },
}

def get_translations():
    """FIX: single helper — call this everywhere instead of repeating 2-line lookup."""
    lang_code = request.cookies.get('language', 'en')
    return TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])

# ─────────────────────────────────────────────────────────────────────────────
# Helper: session-based user (for worker routes)
# ─────────────────────────────────────────────────────────────────────────────
def get_logged_in_user():
    if 'user_id' in session:
        return {'uid': session['user_id'], 'role': session.get('user_role', 'farmer')}
    return None

def is_worker_logged_in():
    user = get_logged_in_user()
    return user and user['role'] == 'worker'

# ─────────────────────────────────────────────────────────────────────────────
# AI Helpers
# ─────────────────────────────────────────────────────────────────────────────
def translate(text, source_lang='auto', target_lang='en'):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f'[Translation Error]: {e}')
        return text

def get_together_ai_response(prompt, model=None, system=None, max_tokens=500):
    """Generic Together AI caller — used by both chatbot and disease fallback."""
    if not TOGETHER_API_KEY:
        return 'AI service not configured.'
    model  = model  or AI_MODEL_CHAT
    system = system or 'You are a helpful assistant for Indian farmers.'
    headers = {
        'Authorization': f'Bearer {TOGETHER_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': prompt},
        ],
        'temperature': 0.7,
        'max_tokens': max_tokens,
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'[Together AI Error]: {e}')
        return 'Sorry, I could not process your request right now.'

def fetch_ai_cause_cure(disease_name):
    prompt = (
        f'A plant is diagnosed with: {disease_name}.\n'
        'Provide:\n1. The likely cause.\n2. A possible cure.\n\nFormat:\nCause: ...\nCure: ...'
    )
    system = 'You are an expert plant pathologist.'
    message = get_together_ai_response(prompt, model=AI_MODEL_DISEASE,
                                       system=system, max_tokens=400)
    if 'Cause:' in message and 'Cure:' in message:
        parts = message.split('Cure:')
        return parts[0].replace('Cause:', '').strip(), parts[1].strip()
    return 'Not specified by AI', 'Not specified by AI'

# ─────────────────────────────────────────────────────────────────────────────
# Plant Disease Detection Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/plant_disease_detection', methods=['GET'])
def plant_disease_detection():
    return render_template('plant_disease_detection.html')

# FIX: was duplicate '/' — now uses its own distinct path
@app.route('/upload_disease_image', methods=['POST'])
def uploadimage():
    if disease_model is None:
        return render_template('plant_disease_detection.html',
                               error='Model loading failed.')
    if 'img' not in request.files:
        return redirect(request.url)
    image = request.files['img']
    if image.filename == '':
        return redirect(request.url)
    filename = f'temp_{uuid.uuid4().hex}_{secure_filename(image.filename)}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER_DISEASE'], filename)
    image.save(filepath)
    prediction_result = model_predict(filepath)
    return render_template(
        'plant_disease_detection.html',
        result=True,
        imagepath=url_for('uploaded_images', filename=filename),
        prediction=prediction_result,
    )

@app.route('/uploadimages/<filename>')
def uploaded_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_DISEASE'], filename)

def extract_features(image_path):
    try:
        img     = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(img)
        return np.expand_dims(feature, axis=0)
    except Exception as e:
        print(f'Error loading image: {e}')
        return None

def model_predict(image_path):
    img_array = extract_features(image_path)
    if img_array is None:
        return {'name': 'Error', 'cause': 'Could not process the uploaded image.', 'cure': ''}
    prediction      = disease_model.predict(img_array)
    max_probability = np.max(prediction)
    if max_probability >= CONFIDENCE_THRESHOLD:
        predicted_label = label[np.argmax(prediction)]
        if predicted_label in plant_disease_info:
            return plant_disease_info[predicted_label]
        cause, cure = fetch_ai_cause_cure(predicted_label)
        return {'name': predicted_label, 'cause': cause, 'cure': cure}
    return {
        'name': 'Unknown Disease',
        'cause': 'Not enough confidence to identify this disease.',
        'cure': 'Please consult an agronomist.',
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main / Home Page
# FIX: no longer conflicts with uploadimage — each has a unique path
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', firestore=firestore,
                           translations=get_translations())

# ─────────────────────────────────────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if not firebase_auth:
        flash('Firebase Auth not initialised.', 'danger')
        return render_template('signup.html', firestore=firestore)
    if request.method == 'POST':
        username = request.form.get('username')
        email    = request.form.get('email')
        password = request.form.get('password')
        try:
            user = firebase_auth.create_user(email=email, password=password,
                                              display_name=username)
            db.collection('users').document(user.uid).set(
                {'username': username, 'email': email}
            )
            flash('Signup Successful! Please Login.', 'success')
            return redirect(url_for('login'))
        except firebase_admin.auth.EmailAlreadyExistsError:
            flash('Email Already Exists', 'warning')
        except Exception as e:
            flash(f'Signup Failed: {e}', 'danger')
    return render_template('signup.html', firestore=firestore)

@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html', firestore=firestore)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout Successful', 'warning')
    return redirect(url_for('login'))

@app.route('/verify_token', methods=['POST'])
def verify_token():
    try:
        data     = request.get_json()
        id_token = data.get('token')
        if not id_token:
            return jsonify({'success': False, 'error': 'No token provided'})
        decoded_token = firebase_auth.verify_id_token(id_token)
        user = User(decoded_token['uid'])
        login_user(user)
        return jsonify({'success': True})
    except Exception as e:
        print('VERIFY TOKEN ERROR:', e)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/create_firebase_user', methods=['POST'])
def create_firebase_user():
    data     = request.get_json()
    id_token = data.get('token')
    username = data.get('username')
    email    = data.get('email')
    if not id_token:
        return jsonify({'success': False, 'error': 'No ID token provided'}), 400
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token.get('uid')
        if uid:
            db.collection('farmers').document(uid).set(
                {'uid': uid, 'username': username, 'email': email}
            )
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid ID token'}), 401
    except firebase_admin.auth.InvalidIdTokenError:
        return jsonify({'success': False, 'error': 'Invalid ID token'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {e}'}), 500

# ─────────────────────────────────────────────────────────────────────────────
# Farmer Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/register', methods=['POST', 'GET'])
@login_required
def register():
    t = get_translations()
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))

    farming_types = [
        {'id': doc.id, 'name': doc.to_dict().get('farmingtype')}
        for doc in db.collection('farming_types').get()
    ]

    if request.method == 'POST':
        farmingtype_id = request.form.get('farming_type')
        try:
            ft_doc = db.collection('farming_types').document(farmingtype_id).get()
            if not ft_doc.exists:
                flash(t.get('invalid_farming_type', 'Invalid farming type.'), 'danger')
                return render_template('register.html', farming_types=farming_types,
                                       firestore=firestore, translations=t)
            db.collection('farmers').add({
                'user_id':     current_user.id,
                'farmername':  request.form.get('farmername'),
                'adharnumber': request.form.get('adharnumber'),
                'age':         int(request.form.get('age') or 0),
                'gender':      request.form.get('gender'),
                'phonenumber': request.form.get('phonenumber'),
                'address':     request.form.get('address'),
                'farming':     ft_doc.to_dict().get('farmingtype'),
            })
            return redirect('/farmerdetails')
        except Exception as e:
            flash(t.get('registration_error', 'Error: {e}').format(e=e), 'danger')

    return render_template('register.html', farming_types=farming_types,
                           firestore=firestore, translations=t)

@app.route('/farmerdetails')
@login_required
def farmerdetails():
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    farmer_data = []
    for doc in db.collection('farmers').where('user_id', '==', current_user.id).get():
        d = doc.to_dict()
        d['id'] = doc.id
        farmer_data.append(d)
    return render_template('farmerdetails.html', query=farmer_data,
                           firestore=firestore, translations=get_translations())

@app.route('/delete/<string:farmer_id>', methods=['POST', 'GET'])
@login_required
def delete(farmer_id):
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    try:
        ref = db.collection('farmers').document(farmer_id)
        doc = ref.get()
        if doc.exists and doc.to_dict().get('user_id') == current_user.id:
            ref.delete()
            flash(f'Farmer {farmer_id} deleted.', 'success')
        elif not doc.exists:
            flash('Farmer not found.', 'warning')
        else:
            flash('Not authorised to delete this farmer.', 'danger')
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return redirect('/farmerdetails')

@app.route('/edit/<string:rid>', methods=['POST', 'GET'])
@login_required
def edit(rid):
    flash('Edit logic needs to be updated for Firestore.', 'warning')
    return render_template('edit.html', posts=None, farming=None, firestore=firestore)

# ─────────────────────────────────────────────────────────────────────────────
# Farming Types
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/addfarming', methods=['POST', 'GET'])
@login_required
def addfarming():
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    if request.method == 'POST':
        farmingtype = request.form.get('farming')
        existing = db.collection('farming_types').where('farmingtype', '==', farmingtype).limit(1).get()
        if existing:
            flash('Farming Type Already Exists', 'warning')
            return redirect('/addfarming')
        db.collection('farming_types').add({'farmingtype': farmingtype})
        flash('Farming Added', 'success')
    return render_template('farming.html', firestore=firestore,
                           translations=get_translations())

# ─────────────────────────────────────────────────────────────────────────────
# Agro Products
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/addagroproducts', methods=['POST', 'GET'])
@login_required
def addagroproducts():
    t = get_translations()
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        username    = request.form.get('username')
        email       = request.form.get('email')
        productname = request.form.get('productname')
        productdesc = request.form.get('productdesc')
        price       = request.form.get('price')
        unit        = request.form.get('unit')

        if not email or not email.endswith('@gmail.com'):
            flash('Please use a valid @gmail.com email address.', 'danger')
            return render_template('addagroproducts.html', firestore=firestore,
                                   translations=t, username=username, email=email,
                                   productname=productname, productdesc=productdesc,
                                   price=price, unit=unit)

        image_filename = None
        file = request.files.get('product_image')
        if file and file.filename:
            if allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    image_filename = filename
                except Exception as e:
                    flash(f'Error uploading image: {e}', 'danger')
            else:
                flash('Invalid file type. Only PNG, JPG, JPEG, GIF are allowed.', 'warning')
                return render_template('addagroproducts.html', firestore=firestore,
                                       translations=t, username=username, email=email,
                                       productname=productname, productdesc=productdesc,
                                       price=price, unit=unit)

        try:
            db.collection('agroproducts').add({
                'user_id':        current_user.id,
                'username':       username,
                'email':          email,
                'productname':    productname,
                'productdesc':    productdesc,
                'price':          float(price) if price else 0.0,
                'unit':           unit,
                'image_filename': image_filename,
            })
            flash('Product Added', 'info')
            return redirect(url_for('agroproducts'))
        except Exception as e:
            flash(f'Error adding product: {e}', 'danger')

    return render_template('addagroproducts.html', firestore=firestore,
                           translations=t)

@app.route('/agroproducts')
def agroproducts():
    t = get_translations()
    if not db:
        flash('Database connection error.', 'danger')
        return render_template('agroproducts.html', query=[], firestore=firestore,
                               translations=t)
    try:
        data = []
        for doc in db.collection('agroproducts').get():
            d = doc.to_dict()
            d['id']      = doc.id
            d['user_id'] = d.get('user_id')
            data.append(d)
        return render_template('agroproducts.html', query=data,
                               firestore=firestore, translations=t)
    except Exception as e:
        flash(f'Error fetching products: {e}', 'danger')
        return render_template('agroproducts.html', query=[], firestore=firestore,
                               translations=t)

@app.route('/deleteagroproduct/<string:product_id>', methods=['POST'])
@login_required
def deleteagroproduct(product_id):
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    try:
        ref = db.collection('agroproducts').document(product_id)
        doc = ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data.get('user_id') == current_user.id:
                img = data.get('image_filename')
                if img:
                    path = os.path.join(app.config['UPLOAD_FOLDER'], img)
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError as e:
                            flash(f'Could not delete image: {img}', 'warning')
                ref.delete()
                flash('Agro product deleted.', 'success')
            else:
                flash('Not authorised to delete this product.', 'danger')
        else:
            flash('Agro product not found.', 'warning')
    except Exception as e:
        flash(f'Error deleting product: {e}', 'danger')
    return redirect(url_for('agroproducts'))

# ─────────────────────────────────────────────────────────────────────────────
# Equipment & Cart
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/equipment')
def equipment():
    t = get_translations()
    if not db:
        flash('Database connection error.', 'danger')
        return render_template('equipment.html', equipment=[], categories=[],
                               brands=[], current_category=None, current_brand=None,
                               firestore=firestore, translations=t)

    categories, brands, equipment_data = set(), set(), []
    for doc in db.collection('equipment').get():
        d = doc.to_dict(); d['id'] = doc.id
        equipment_data.append(d)
        if 'category' in d: categories.add(d['category'])
        if 'brand'    in d: brands.add(d['brand'])

    cat_f   = request.args.get('category')
    brand_f = request.args.get('brand')
    filtered = [
        e for e in equipment_data
        if (not cat_f   or e.get('category') == cat_f)
        and (not brand_f or e.get('brand')    == brand_f)
    ]
    return render_template('equipment.html', equipment=filtered, categories=categories,
                           brands=brands, current_category=cat_f, current_brand=brand_f,
                           firestore=firestore, translations=t)

@app.route('/equipment/add_to_cart', methods=['POST'])
@login_required
def add_to_cart():
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))

    equipment_id  = request.form['equipment_id']
    equipment_ref = db.collection('equipment').document(equipment_id)
    equipment_doc = equipment_ref.get()

    if not equipment_doc.exists:
        flash('Equipment not found.', 'danger')
        return redirect('/equipment')

    data  = equipment_doc.to_dict()
    price = data.get('price')
    if price is None or not isinstance(price, (int, float)):
        flash(f"{data.get('name')} has an invalid price.", 'warning')
        return redirect('/equipment')

    try:
        available_quantity = int(data.get('available_quantity', 0))
    except ValueError:
        flash(f"{data.get('name')} has an invalid available quantity.", 'warning')
        return redirect('/equipment')

    if available_quantity <= 0:
        flash(f"{data.get('name')} is out of stock.", 'warning')
        return redirect('/equipment')

    cart_ref   = db.collection('users').document(current_user.id).collection('cart')
    cart_query = [i for i in cart_ref.where('equipment_id', '==', equipment_id).limit(1).get()]

    if cart_query:
        item_ref  = cart_ref.document(cart_query[0].id)
        new_qty   = cart_query[0].to_dict().get('quantity', 0) + 1
        if new_qty <= available_quantity:
            item_ref.update({'quantity': new_qty, 'price': price})
            equipment_ref.update({'available_quantity': available_quantity - 1})
            flash(f"Added one {data.get('name')} to cart.", 'success')
        else:
            flash(f"Not enough {data.get('name')} in stock.", 'warning')
    else:
        cart_ref.add({'equipment_id': equipment_id, 'name': data.get('name'),
                      'price': price, 'quantity': 1})
        equipment_ref.update({'available_quantity': available_quantity - 1})
        flash(f"Added {data.get('name')} to cart.", 'success')

    return redirect('/equipment')

@app.route('/cart')
@login_required
def cart():
    t = get_translations()
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    try:
        cart_data, total = [], 0
        for item in db.collection('users').document(current_user.id).collection('cart').get():
            d = item.to_dict(); d['id'] = item.id
            cart_data.append(d)
            total += d.get('price', 0) * d.get('quantity', 0)
        return render_template('cart.html', cart_items=cart_data, total_amount=total,
                               firestore=firestore, translations=t)
    except Exception:
        flash('Error loading cart.', 'danger')
        return render_template('cart.html', cart_items=[], total_amount=0,
                               firestore=firestore, translations=t)

@app.route('/cart/remove/<item_id>', methods=['POST'])
@login_required
def remove_from_cart(item_id):
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    db.collection('users').document(current_user.id).collection('cart').document(item_id).delete()
    flash('Item removed from cart.', 'info')
    return redirect('/cart')

@app.route('/cart/update/<item_id>', methods=['POST'])
@login_required
def update_cart_quantity(item_id):
    if not (db and current_user.is_authenticated):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login'))
    try:
        new_qty = int(request.form['quantity'])
        if new_qty < 1:
            flash('Quantity must be at least 1.', 'warning')
            return redirect('/cart')
        cart_ref  = db.collection('users').document(current_user.id).collection('cart')
        item_ref  = cart_ref.document(item_id)
        item_doc  = item_ref.get()
        if not item_doc.exists:
            flash('Cart item not found.', 'danger')
            return redirect('/cart')
        item_data     = item_doc.to_dict()
        old_qty       = item_data.get('quantity', 0)
        qty_diff      = new_qty - old_qty
        equipment_ref = db.collection('equipment').document(item_data.get('equipment_id'))
        equipment_doc = equipment_ref.get()
        if not equipment_doc.exists:
            flash('Equipment not found.', 'danger')
            return redirect('/cart')
        stock = equipment_doc.to_dict().get('available_quantity', 0)
        if qty_diff > 0 and stock < qty_diff:
            flash('Not enough stock to increase quantity.', 'warning')
            return redirect('/cart')
        equipment_ref.update({'available_quantity': stock - qty_diff})
        item_ref.update({'quantity': new_qty})
        flash('Cart updated.', 'success')
    except Exception as e:
        print(f'Update error: {e}')
        flash('Error updating cart.', 'danger')
    return redirect('/cart')

# ─────────────────────────────────────────────────────────────────────────────
# Crop Recommendation
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/preone')
def preone():
    return render_template('preone.html', firestore=firestore)

@app.route('/predict', methods=['POST'])
def predict():
    if not all([crop_model, crop_sc, crop_mx]):
        return render_template('preone.html', result='Crop prediction models are not loaded.',
                               firestore=firestore)
    try:
        features = [
            float(request.form['Nitrogen']),   float(request.form['Phosporus']),
            float(request.form['Potassium']),  float(request.form['Temperature']),
            float(request.form['Humidity']),   float(request.form['pH']),
            float(request.form['Rainfall']),
        ]
        arr        = np.array(features).reshape(1, -1)
        scaled     = crop_sc.transform(crop_mx.transform(arr))
        prediction = crop_model.predict(scaled)
        crop_dict  = {
            1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya',
            7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes',
            12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil', 16: 'Blackgram',
            17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas', 20: 'Kidneybeans',
            21: 'Chickpea', 22: 'Coffee',
        }
        crop   = crop_dict.get(prediction[0], '')
        result = f'{crop} is the best crop to be cultivated right there' if crop else \
                 'Sorry, we could not determine the best crop.'
        return render_template('preone.html', result=result, firestore=firestore)
    except (ValueError, KeyError):
        return render_template('preone.html', result='Invalid input.', firestore=firestore)
    except Exception as e:
        return render_template('preone.html', result=f'Unexpected error: {e}',
                               firestore=firestore)

# ─────────────────────────────────────────────────────────────────────────────
# AI Chatbot
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/chat')
@login_required
def chat():
    user_id = current_user.id
    chats   = []
    for doc in db.collection('users').document(user_id).collection('chats') \
                 .order_by('timestamp', direction=firestore.Query.DESCENDING).stream():
        d = doc.to_dict()
        chats.append({'id': doc.id, 'title': d.get('title', 'Untitled')})
    return render_template('chat.html', chats=chats)

@app.route('/chat/<chat_id>')
@login_required
def view_chat(chat_id):
    user_id  = current_user.id
    all_chats = []
    for doc in db.collection('users').document(user_id).collection('chats') \
                 .order_by('timestamp', direction=firestore.Query.DESCENDING).stream():
        d = doc.to_dict()
        all_chats.append({'id': doc.id, 'title': d.get('title', 'Untitled')})

    chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_id)
    snap     = chat_ref.get()
    if snap.exists:
        messages = [
            {'role': m.to_dict()['role'], 'text': m.to_dict()['text']}
            for m in chat_ref.collection('messages').order_by('timestamp').stream()
        ]
        return render_template('chat.html', messages=messages, chats=all_chats,
                               current_chat_id=chat_id)
    flash('Chat not found.')
    return redirect(url_for('chat'))

@app.route('/chat', methods=['POST'])
@login_required
def chat_response():
    data              = request.get_json()
    user_input        = data.get('text', '')
    user_lang         = data.get('lang', 'en')
    user_id           = current_user.id
    chat_id_from_req  = data.get('chat_id')
    is_new            = not bool(chat_id_from_req)
    current_chat_id   = chat_id_from_req or datetime.utcnow().strftime('%Y%m%d%H%M%S%f')

    question_en  = translate(user_input, source_lang='auto', target_lang='en')
    answer_en    = get_together_ai_response(question_en)
    answer_local = translate(answer_en, source_lang='en', target_lang=user_lang)

    save_chat_message(user_id, current_chat_id, 'user', user_input, is_first_message=is_new)
    save_chat_message(user_id, current_chat_id, 'bot',  answer_local)

    return jsonify({'response': answer_local, 'chat_id': current_chat_id})

def save_chat_message(user_id, chat_id, role, message, is_first_message=False):
    chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_id)
    chat_doc = chat_ref.get()
    if not chat_doc.exists:
        title = message[:30] if role == 'user' else 'Chat Session'
        chat_ref.set({'title': title, 'timestamp': datetime.utcnow()})
    elif is_first_message and role == 'user':
        chat_ref.update({'title': message[:30]})
    chat_ref.collection('messages').add({
        'role': role, 'text': message, 'timestamp': datetime.utcnow()
    })

@app.route('/get_chats/<user_id>', methods=['GET'])
def get_chats(user_id):
    result = []
    for doc in db.collection('users').document(user_id).collection('chats') \
                 .order_by('timestamp', direction=firestore.Query.DESCENDING).stream():
        d = doc.to_dict()
        result.append({'id': doc.id, 'title': d.get('title', 'Untitled'),
                       'timestamp': d.get('timestamp')})
    return jsonify(result)

@app.route('/get_messages/<user_id>/<chat_id>', methods=['GET'])
def get_messages(user_id, chat_id):
    msgs = db.collection('users').document(user_id).collection('chats') \
             .document(chat_id).collection('messages').order_by('timestamp').stream()
    return jsonify([{'role': m.to_dict()['role'], 'text': m.to_dict()['text']} for m in msgs])

# ─────────────────────────────────────────────────────────────────────────────
# Worker Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/worker')
def worker_home():
    workers_data = []
    try:
        for doc in db.collection('workers').stream():
            w = doc.to_dict(); w['id'] = doc.id
            workers_data.append(w)
    except Exception as e:
        flash(f'Error fetching workers: {e}', 'error')
    return render_template('worker.html', workers=workers_data,
                           current_user=get_logged_in_user())

@app.route('/signup1.html')
def worker_signup():
    return render_template('signup1.html')

@app.route('/worker/signup', methods=['POST'])
def worker_signup_post():
    username         = request.form.get('username')
    email            = request.form.get('email')
    password         = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if not all([username, email, password, confirm_password]):
        flash('All fields are required.', 'error')
        return render_template('signup1.html', username=username, email=email)

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return render_template('signup1.html', username=username, email=email)

    users_ref = db.collection('users')
    if list(users_ref.where('username', '==', username).limit(1).stream()):
        flash('Username already taken.', 'error')
        return render_template('signup1.html', username=username, email=email)
    if list(users_ref.where('email', '==', email).limit(1).stream()):
        flash('Email already registered.', 'error')
        return render_template('signup1.html', username=username, email=email)

    # FIX: password is hashed before storing
    hashed_pw  = generate_password_hash(password)
    user_data  = {'username': username, 'email': email,
                  'password': hashed_pw, 'role': 'worker'}
    try:
        new_ref = users_ref.document()
        new_ref.set(user_data)
        session['user_id']   = new_ref.id
        session['user_role'] = 'worker'
        flash('Signup successful! Please complete your worker profile.', 'success')
        return redirect(url_for('worker_dashboard'))
    except Exception as e:
        flash(f'An error occurred: {e}', 'error')
        return render_template('signup1.html', username=username, email=email)

@app.route('/login1.html')
def worker_login():
    return render_template('login1.html')

@app.route('/worker/login', methods=['POST'])
def worker_login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    if not username or not password:
        flash('Username and password are required.', 'error')
        return render_template('login1.html', username=username)

    users_ref  = db.collection('users')
    # FIX: only query by username — verify hashed password in Python
    user_list  = list(users_ref.where('username', '==', username).limit(1).stream())

    if user_list:
        user_doc  = user_list[0]
        user_data = user_doc.to_dict()
        # FIX: use check_password_hash to compare securely
        if check_password_hash(user_data.get('password', ''), password):
            role = user_data.get('role', 'farmer')
            if role == 'worker':
                session['user_id']   = user_doc.id
                session['user_role'] = role
                flash('Login successful!', 'success')
                return redirect(url_for('worker_dashboard'))
            flash('These credentials are not for a worker account.', 'error')
            return render_template('login1.html', username=username)

    flash('Invalid username or password.', 'error')
    return render_template('login1.html', username=username)

@app.route('/worker/dashboard')
def worker_dashboard():
    if not is_worker_logged_in():
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('worker_login'))
    user_id     = session['user_id']
    worker_list = list(db.collection('workers').where('userId', '==', user_id).limit(1).stream())
    worker_data = None
    if worker_list:
        worker_data = worker_list[0].to_dict()
        worker_data['doc_id'] = worker_list[0].id
    return render_template('worker1.html', worker_data=worker_data,
                           current_user=get_logged_in_user())

@app.route('/worker/register', methods=['POST'])
def register_worker():
    if not is_worker_logged_in():
        return redirect(url_for('worker_login'))
    user_id     = session['user_id']
    worker_data = _build_worker_data(user_id)
    workers_ref = db.collection('workers')
    existing    = list(workers_ref.where('userId', '==', user_id).limit(1).get())
    if existing:
        workers_ref.document(existing[0].id).update(worker_data)
    else:
        workers_ref.add(worker_data)
    return redirect(url_for('worker_dashboard'))

@app.route('/worker/update', methods=['POST'])
def update_worker_profile():
    if not is_worker_logged_in():
        return redirect(url_for('worker_login'))
    user_id  = session['user_id']
    data     = _build_worker_data(user_id)
    existing = list(db.collection('workers').where('userId', '==', user_id).limit(1).get())
    if existing:
        db.collection('workers').document(existing[0].id).update(data)
    return redirect(url_for('worker_dashboard'))

@app.route('/worker/profile/save', methods=['POST'])
def save_worker_profile():
    if not is_worker_logged_in():
        flash('Authentication required.', 'error')
        return redirect(url_for('worker_login'))

    user_id             = session['user_id']
    contact_number_str  = request.form.get('contact_number', '').strip()
    name                = request.form.get('name')
    experience          = request.form.get('experience')
    per_day_price_str   = request.form.get('per_day_price')
    expertise           = request.form.get('expertise')

    if not re.match(r'^\d{10}$', contact_number_str):
        flash('Contact number must be exactly 10 digits.', 'error')
        return redirect(url_for('worker_dashboard'))

    if not all([name, experience, per_day_price_str, expertise]):
        flash('Please fill in all required fields.', 'error')
        return redirect(url_for('worker_dashboard'))

    image_db_filename = None
    if 'profile_image' in request.files:
        file = request.files['profile_image']
        if file and file.filename and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_db_filename = filename
            except Exception as e:
                flash(f'Could not save image: {e}', 'error')
        elif file.filename and not allowed_file(file.filename):
            flash('Invalid image file type.', 'error')
            return redirect(url_for('worker_dashboard'))

    profile = {
        'userId':        user_id,
        'name':          name,
        'experience':    experience,
        'workDetails':   request.form.get('work_details'),
        'perDayPrice':   float(per_day_price_str) if per_day_price_str else 0.0,
        'expertise':     expertise,
        'availability':  request.form.get('availability', 'Available'),
        'contactNumber': contact_number_str,
    }

    workers_ref = db.collection('workers')
    existing    = list(workers_ref.where('userId', '==', user_id).limit(1).stream())
    try:
        if existing:
            existing_data = existing[0].to_dict()
            profile['imageUrl'] = image_db_filename or existing_data.get('imageUrl')
            workers_ref.document(existing[0].id).update(profile)
            flash('Profile updated successfully!', 'success')
        else:
            profile['imageUrl'] = image_db_filename
            workers_ref.add(profile)
            flash('Profile registered successfully!', 'success')
    except Exception as e:
        flash(f'Error saving profile: {e}', 'error')
    return redirect(url_for('worker_dashboard'))

def _build_worker_data(user_id):
    """Build worker dict from form — avoids duplicate form-parsing code."""
    per_day  = request.form.get('per_day_price')
    contact  = request.form.get('contact_number')
    return {
        'userId':        user_id,
        'name':          request.form.get('name'),
        'imageUrl':      request.form.get('image_url'),
        'experience':    request.form.get('experience'),
        'workDetails':   request.form.get('work_details'),
        'perDayPrice':   float(per_day) if per_day else 0.0,
        'expertise':     request.form.get('expertise'),
        'availability':  request.form.get('availability', 'Available'),
        'contactNumber': contact or '',
    }

@app.route('/worker/logout')
def worker_logout():
    session.pop('user_id',   None)
    session.pop('user_role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('worker_home'))

# ─────────────────────────────────────────────────────────────────────────────
# Static / Misc Pages
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/govschems')
def govschems():
    return render_template('govschems.html', firestore=firestore,
                           translations=get_translations())

@app.route('/Plantmood')
def Plantmood():
    return render_template('Plantmood.html', firestore=firestore,
                           translations=get_translations())

@app.route('/map')
def map():
    return render_template('map.html', firestore=firestore)

@app.route('/transport')
def transport():
    return render_template('transport.html', firestore=firestore,
                           translations=get_translations())

@app.route('/croproadmap')
def croproadmap():
    return render_template('croproadmap.html', firestore=firestore)

# FIX: decorator was merged into a comment in the original — now properly registered
@app.route('/test')
def test():
    if db:
        try:
            docs = db.collection('agroproducts').get()
            return f'Firestore connected. Found {len(docs)} documents in agroproducts.'
        except Exception as e:
            return f'Error connecting to Firestore: {e}'
    return 'Firestore not initialised.'

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)