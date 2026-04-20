from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_from_directory
from markupsafe import Markup
from werkzeug.utils import secure_filename
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
from firebase_admin import credentials, firestore, auth, exceptions, initialize_app
from deep_translator import GoogleTranslator
from datetime import datetime
from flask import request, jsonify, session
from firebase_admin import auth as firebase_auth


# Load the machine learning models for crop recommendation , soil test
try:
    model1 = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    print("ML models loaded successfully!")
except Exception as e:
    print(f"Error loading ML models: {e}")
    model1, sc, mx = None, None, None

# Initialize Firebase Admin SDK (database connection)
# Firebase Admin Initialization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(BASE_DIR, "secrets", "firebase_key.json")

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_auth = auth
        print("Firebase connected successfully!")
except Exception as e:
    print("Firebase init error:", e)

app = Flask(__name__)
app.secret_key = 'harshithbhaskar'



# API key and URL for Together AI
TOGETHER_API_KEY = "api key"  # Replace with your actual API key
TOGETHER_API_URL = 'https://api.together.xyz/v1/chat/completions'
AI_MODEL = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'  # Or the model you prefer




#####################################################################################################################
TOGETHER_API_KEY1 = "api key"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
AI_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"




# plant disease detection model code

MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"  # Use relative path
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle the case where the model fails to load



label = ['Apple___Apple_scab',
         'Apple___Black_rot',
         'Apple___Cedar_apple_rust',
         'Apple___healthy',
         'Background_without_leaves',
         'Blueberry___healthy',
         'Cherry___Powdery_mildew',
         'Cherry___healthy',
         'Corn___Cercospora_leaf_spot Gray_leaf_spot',
         'Corn___Common_rust',
         'Corn___Northern_Leaf_Blight',
         'Corn___healthy',
         'Grape___Black_rot',
         'Grape___Esca_(Black_Measles)',
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy',
         'Orange___Haunglongbing_(Citrus_greening)',
         'Peach___Bacterial_spot',
         'Peach___healthy',
         'Pepper,_bell___Bacterial_spot',
         'Pepper,_bell___healthy',
         'Potato___Early_blight',
         'Potato___Late_blight',
         'Potato___healthy',
         'Raspberry___healthy',
         'Soybean___healthy',
         'Squash___Powdery_mildew',
         'Strawberry___Leaf_scorch',
         'Strawberry___healthy',
         'Tomato___Bacterial_spot',
         'Tomato___Early_blight',
         'Tomato___Late_blight',
         'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite',
         'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus',
         'Tomato___healthy']

with open("plant_disease.json", 'r') as file:
    plant_disease_info = json.load(file)

UPLOAD_FOLDER = 'uploadimages'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Confidence threshold for considering a prediction valid
CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed

@app.route('/plant_disease_detection', methods=['GET'])
def plant_disease_detection():
    return render_template('plant_disease_detection.html')


@app.route('/', methods=['POST'])
def uploadimage():
    if model is None:
        return render_template('plant_disease_detection.html', error="Model loading failed.")

    if 'img' not in request.files:
        return redirect(request.url)
    image = request.files['img']
    if image.filename == '':
        return redirect(request.url)
    if image:
        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        print(f'Saved image to: {filepath}')
        prediction_result = model_predict(filepath)
        return render_template('plant_disease_detection.html', result=True, imagepath=url_for('uploaded_images', filename=filename), prediction=prediction_result)
    return redirect(request.url)

@app.route('/uploadimages/<filename>')
def uploaded_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def extract_features(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(img)
        feature = np.expand_dims(feature, axis=0)
        return feature
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def model_predict(image_path):
    img_array = extract_features(image_path)
    if img_array is None:
        return {
            "name": "Error",
            "cause": "Could not process the uploaded image.",
            "cure": ""
        }

    prediction = model.predict(img_array)
    max_probability = np.max(prediction)

    if max_probability >= CONFIDENCE_THRESHOLD:
        predicted_class_index = np.argmax(prediction)
        predicted_label = label[predicted_class_index]

        if predicted_label in plant_disease_info:
            return plant_disease_info[predicted_label]
        else:
            cause, cure = fetch_ai_cause_cure(predicted_label)
            return {
                "name": predicted_label,
                "cause": cause,
                "cure": cure
            }
    else:
        return {
            "name": "Unknown Disease",
            "cause": "I don't have enough confidence to identify this disease.",
            "cure": "I don't have enough information."
        }


def fetch_ai_cause_cure(disease_name):
    prompt = f"""
    A plant is diagnosed with the disease: {disease_name}.
    Please provide:
    1. The likely cause of this disease.
    2. A possible cure or treatment.

    Respond in this format:
    Cause: ...
    Cure: ...
    """

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY1}",
        "Content-Type": "application/json"
    }

    body = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert plant pathologist."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(TOGETHER_API_URL, json=body, headers=headers)
        if response.status_code == 200:
            content = response.json()
            message = content['choices'][0]['message']['content']
            if "Cause:" in message and "Cure:" in message:
                parts = message.split("Cure:")
                cause = parts[0].replace("Cause:", "").strip()
                cure = parts[1].strip()
                return cause, cure
            else:
                return "Not specified by AI", "Not specified by AI"
        else:
            print("AI API call failed:", response.text)
            return "Unable to retrieve cause", "Unable to retrieve cure"
    except Exception as e:
        print(f"Error calling Together AI: {e}")
        return "AI error", "AI error"
#######################################################################################################



# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, uid):
        self.id = uid


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)




###################################################################################################



#AI Chat bot
# Initialize Firebase

# cred_path = r"C:\Users\mkesh\OneDrive\Desktop\Projects\Final Year Project\KrishiJyothi\app\secrets\farm-management-system-a1acd-7138a6de8031.json"
# cred = credentials.Certificate(cred_path)

# if not firebase_admin._apps:
#     firebase_admin.initialize_app(cred)
# else:
#     print("Firebase app already initialized.")
# db = firestore.client()

TOGETHER_API_KEY = "api key"

def translate(text, source_lang="auto", target_lang="en"):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        print(f"[Translation Error]: {e}")
        return text

def get_together_ai_response(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for Indian farmers. Answer in a clear and simple way."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Together AI Error]: {e}")
        return "Sorry, I couldn't process your request right now."

# Chat page with chat history sidebar
@app.route("/chat")
@login_required
def chat():
    user_id = current_user.id
    chats = []
    chat_docs = db.collection("users").document(user_id).collection("chats") \
                   .order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    for doc in chat_docs:
        data = doc.to_dict()
        chats.append({
            "id": doc.id,
            "title": data.get("title", "Untitled")
        })
    return render_template("chat.html", chats=chats)


# View chat by chat_id, show messages
@app.route("/chat/<chat_id>")
@login_required
def view_chat(chat_id):
    user_id = current_user.id
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
    chat_doc_snapshot = chat_ref.get() # Renamed to avoid conflict with 'chat' variable name for list

    # Fetch all chats for the sidebar (same logic as in the main /chat route)
    all_chats_list = []
    chat_docs_sidebar = db.collection("users").document(user_id).collection("chats") \
                        .order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    for doc_sidebar in chat_docs_sidebar:
        data_sidebar = doc_sidebar.to_dict()
        all_chats_list.append({
            "id": doc_sidebar.id,
            "title": data_sidebar.get("title", "Untitled")
        })

    if chat_doc_snapshot.exists:
        # data = chat_doc_snapshot.to_dict() # Not strictly needed here if just getting messages
        messages_ref = chat_ref.collection("messages").order_by("timestamp").stream()
        messages = [{'role': m.to_dict()['role'], 'text': m.to_dict()['text']} for m in messages_ref]

        # Pass both all_chats_list for the sidebar and messages for the current chat
        return render_template("chat.html", messages=messages, chats=all_chats_list, current_chat_id=chat_id) # Changed current_chat to current_chat_id
    else:
        flash("Chat not found.")
        return redirect(url_for("chat"))

# Chatbot API: receive user input and respond
@app.route("/chat", methods=["POST"])
@login_required
def chat_response():
    data = request.get_json()
    user_input = data.get("text", "")
    user_lang = data.get("lang", "en")
    user_id = current_user.id

    # Get chat_id from the request. If it's not there, this is a new chat.
    chat_id_from_request = data.get("chat_id")
    
    is_new_chat_session = not bool(chat_id_from_request) # True if chat_id_from_request is None or empty

    if is_new_chat_session:
        # Generate a new chat_id if one wasn't provided by the client
        current_chat_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        print(f"\nNew chat session started by User ID: {user_id}. Generated Chat ID: {current_chat_id}")
    else:
        current_chat_id = chat_id_from_request
        print(f"\nContinuing chat session for User ID: {user_id}. Chat ID: {current_chat_id}")

    print(f"User question: {user_input} | Language: {user_lang}")

    question_en = translate(user_input, source_lang="auto", target_lang="en")
    answer_en = get_together_ai_response(question_en)
    answer_local = translate(answer_en, source_lang="en", target_lang=user_lang)

    # Save user message. Pass is_new_chat_session hint for title setting.
    save_chat_message(user_id, current_chat_id, "user", user_input, is_first_message=is_new_chat_session)
    # Save bot message. This is never the "first" message for title setting purposes.
    save_chat_message(user_id, current_chat_id, "bot", answer_local, is_first_message=False)

    return jsonify({"response": answer_local, "chat_id": current_chat_id})
# Save messages in subcollection 'messages'


# Save messages in subcollection 'messages'
def save_chat_message(user_id, chat_id, role, message, is_first_message=False):
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
    chat_doc = chat_ref.get()

    if not chat_doc.exists:
        # This is the very first time any message is saved for this chat_id,
        # so the chat document itself needs to be created.
        title_to_set = "Chat Session" # Default title
        if role == "user": # If the first message ever for this chat_id is from the user
            title_to_set = message[:30]
        
        chat_ref.set({
            "title": title_to_set,
            "timestamp": datetime.utcnow() # Timestamp for the chat session creation
        })
    elif is_first_message and role == "user" and not chat_doc.to_dict().get("title_set_by_user"):
        # This condition can be used if you want to ensure the title is set by the *actual first user message of a session*,
        # even if the chat_doc was technically created by some other means or an earlier bot message (less likely with current flow).
        # For simplicity, the `if not chat_doc.exists:` often covers this.
        # Adding 'title_set_by_user' flag can prevent overwriting a title if it was already set.
        # However, the current logic in chat_response (is_new_chat_session) combined with
        # `if not chat_doc.exists` should suffice.
        # If you want to explicitly update if it's the first *user* message of a *newly identified session*:
        chat_ref.update({
             "title": message[:30],
        #     "title_set_by_user": True # Optional: flag to prevent future updates
        })


    # Add message as a document in messages subcollection
    chat_ref.collection("messages").add({
        "role": role,
        "text": message,
        "timestamp": datetime.utcnow() # Timestamp for the individual message
    })

# Additional APIs for frontend if needed
@app.route('/get_chats/<user_id>', methods=['GET'])
def get_chats(user_id):
    chats_ref = db.collection('users').document(user_id).collection('chats')
    chats = chats_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    result = []
    for chat in chats:
        chat_data = chat.to_dict()
        result.append({
            'id': chat.id,
            'title': chat_data.get('title', 'Untitled'),
            'timestamp': chat_data.get('timestamp')
        })
    return jsonify(result)

@app.route('/get_messages/<user_id>/<chat_id>', methods=['GET'])
def get_messages(user_id, chat_id):
    messages_ref = db.collection('users').document(user_id).collection('chats') \
        .document(chat_id).collection('messages').order_by('timestamp')
    messages = messages_ref.stream()
    result = [{'role': m.to_dict()['role'], 'text': m.to_dict()['text']} for m in messages]
    return jsonify(result)



##########################################################################################################################




#######################################################################################################################
#Website Language change json 
TRANSLATIONS = {
    'en': {
        'home': 'Home',
        'agro_products': 'Agro Products',
        'ai_help': 'AI Help',
        'gov_schems' : 'Government Schemes',
        'hire_labour': 'Hire Labour',
        'soil_test': 'Soil Test',
        'equipment_rental': 'Equipment Rental',
        'more': 'More',
        'smart_agriculture': 'Smart Agriculture',
        'farmer_register': 'Farmer Register',
        'add_farming': 'Add Farming', # Added keys from base.html example
        'farmer_details': 'Farmer Details',
        'smart_farm': 'Smart Farm',
        'map': 'Map',
        'live_prices': 'Live Prices',
        'transport': 'Transport',
        'disease': 'Disease',
        'plant_condition': 'Plant Condition',
        'cart': 'Cart',
        'crop_roadmap': 'Crop Road Map',
        'welcome': 'Welcome', # For the welcome message
        'logout': 'Logout',
        'signup': 'Signup',
        'main_title': 'Smart Agriculture', # For the main intro title
        'intro_subtitle': 'Revolutionizing Farming with Technology', # For the intro subtitle
        'agro_products_button': 'AGRO PRODUCTS', # For the button text
         'plant_disease_detection': 'Plant Disease Detection',
         'fertilizer_prediction': 'Fertiliser Prediction',
          'farmer_name': 'Farmer Name',
        'aadhaar_number': 'Aadhaar Number',
        'age': 'Age',
        'gender': 'Gender',
        'male': 'Male',
        'phone_number': 'Phone Number',
        'address': 'Address',
        'farming_type': 'Farming Type',
        'farming': 'FARMING',
        'register_farmer': 'Register Farmer',
        'farmer_registration': 'Farmer Registration',
          'add': 'ADD',
        'delete_confirmation': 'Are you sure to Delete data',
        'rid': 'RID', 
        'delete': 'DELETE',
         'add_agro_product': 'ADD AGRO PRODUCT',
          'add_agro_products_title': 'Add AgroProducts',
        'add_agro_products_heading': 'Add Agro Products',
        'farmer_name': 'Farmer Name',
        'farmer_email': 'Farmer Email',
        'farmer_email_placeholder': 'yourname@gmail.com',
        'product_name': 'Product Name',
        'upload_product_image': 'Upload Product Image',
        'choose_file': 'Choose File',
        'no_file_chosen': 'No file chosen',
        'image_upload_optional': 'Select an image file (JPG, PNG, etc.). Optional.',
        'product_description': 'Product Description',
        'price': 'Price',
        'unit': 'Unit',
        'kg': 'kg',
        'quintal': 'quintal',
        'add_product_button': 'Add Product',
         'farmer_services': 'Farmer Services',
        'resources': 'Resources',
        'ai_tools': 'AI Tools' # The title you asked for
# Add ALL text strings from your base.html and other templates here
    },
    'hi': {
        'home': 'होम',
        'agro_products': 'कृषि उत्पाद',
        'gov_schems' : 'सरकारी योजनाएं',
        'ai_help': 'एआई मदद',
        'hire_labour': 'श्रमिक किराए पर लें',
        'soil_test': 'मिट्टी परीक्षण',
        'equipment_rental': 'उपकरण किराये',
        'more': 'और अधिक',
        'smart_agriculture': 'स्मार्ट कृषि',
        'farmer_register': 'किसान पंजीकरण',
        'add_farming': 'खेती जोड़ें',
        'farmer_details': 'किसान विवरण',
        'smart_farm': 'स्मार्ट खेत',
        'map': 'नक्शा',
        'live_prices': 'लाइव कीमतें',
        'transport': 'परिवहन',
        'disease': 'रोग',
        'plant_condition': 'पौधे की स्थिति',
        'cart': 'कार्ट',
        'crop_roadmap': 'फसल रोड मैप',
        'welcome': 'स्वागत है',
        'logout': 'लॉग आउट',
        'signup': 'साइन अप करें',
        'main_title': 'स्मार्ट कृषि',
        'intro_subtitle': 'प्रौद्योगिकी से खेती में क्रांति',
        'agro_products_button': 'कृषि उत्पाद', # Button text might also need translation
        'plant_disease_detection': 'पौधे की बीमारी का पता लगाना',
        'fertilizer_prediction': 'उर्वरक पूर्वानुमान',
        'aadhaar_number': 'आधार नंबर',
        'age': 'आयु',
        'gender': 'लिंग',
        'male': 'पुरुष',
        'phone_number': 'फ़ोन नंबर',
        'address': 'पता',
        'farming_type': 'खेती का प्रकार',
        'register_farmer': 'किसान को पंजीकृत करें',
        'farmer_registration': 'किसान पंजीकरण', 
        'add': 'जोड़ें',
        'delete_confirmation': 'क्या आप डेटा हटाना चाहते हैं?',
          'delete': 'हटाएं',
          'rid': 'आरआईडी',
           'farming': 'खेती',
           'add_agro_products_title': 'कृषि उत्पाद जोड़ें',
        'add_agro_products_heading': 'कृषि उत्पाद जोड़ें',
        'farmer_name': 'किसान का नाम',
        'farmer_email': 'किसान ईमेल',
        'farmer_email_placeholder': 'yourname@gmail.com',
        'product_name': 'उत्पाद का नाम',
        'upload_product_image': 'उत्पाद की छवि अपलोड करें',
        'choose_file': 'फ़ाइल चुनें',
        'no_file_chosen': 'कोई फ़ाइल नहीं चुनी गई',
        'image_upload_optional': 'एक छवि फ़ाइल चुनें (JPG, PNG, आदि)। वैकल्पिक।',
        'product_description': 'उत्पाद विवरण',
        'price': 'मूल्य',
        'unit': 'इकाई',
        'kg': 'किग्रा',
        'quintal': 'क्विंटल',
        'add_product_button': 'उत्पाद जोड़ें',
        'farmer_services': 'किसान सेवाएं',
        'resources': 'संसाधन',
        'ai_tools': 'एआई उपकरण',# Add Hindi translations for all keys
    },
    'kn': {
        'home': 'ಹೋಮ್',
        'agro_products': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳು',
        'gov_schems' : 'ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು',
        'ai_help': 'ಎಐ ಸಹಾಯ',
        'hire_labour': 'ಕಾರ್ಮಿಕರನ್ನು ನೇಮಿಸಿಕೊಳ್ಳಿ',
        'soil_test': 'ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ',
        'equipment_rental': 'ಉಪಕರಣ ಬಾಡಿಗೆ',
        'more': 'ಇನ್ನಷ್ಟು',
        'smart_agriculture': 'ಸ್ಮಾರ್ಟ್ ಕೃಷಿ',
        'farmer_register': 'ರೈತ ನೋಂದಣಿ',
        'add_farming': 'ಕೃಷಿ ಸೇರಿಸಿ',
        'farmer_details': 'ರೈತರ ವಿವರಗಳು',
        'smart_farm': 'ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮ್',
        'map': 'ನಕ್ಷೆ',
        'live_prices': 'ಲೈವ್ ಬೆಲೆಗಳು',
        'transport': 'ಸಾರಿಗೆ',
        'disease': 'ರೋಗ',
        'plant_condition': 'ಸಸ್ಯ ಸ್ಥಿತಿ',
        'cart': 'ಕಾರ್ಟ್',
        'crop_roadmap': 'ಬೆಳೆ ಮಾರ್ಗ ನಕ್ಷೆ',
        'welcome': 'ಸ್ವಾಗತ',
        'logout': 'ಲಾಗ್ ಔಟ್',
        'signup': 'ಸೈನ್ ಅಪ್',
        'main_title': 'ಸ್ಮಾರ್ಟ್ ಕೃಷಿ',
        'intro_subtitle': 'ತಂತ್ರಜ್ಞಾನದಿಂದ ಕೃಷಿಯಲ್ಲಿ ಕ್ರಾಂತಿ',
        'agro_products_button': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳು', # Button text might also need translation
        'plant_disease_detection': 'ಮೂಲಕ ಬೆಳೆ ರೋಗ ಪತ್ತೆ',
        'fertilizer_prediction': 'ರಸಗೊಬ್ಬರ ಮುನ್ಸೂಚನೆ',
        'rid': 'RID',
        'farmer_name': 'ರೈತರ ಹೆಸರು',
        'aadhaar_number': 'ಆಧಾರ್ ಸಂಖ್ಯೆ',
        'age': 'ವಯಸ್ಸು',
        'gender': 'ಲಿಂಗ',
        'male': 'ಪುರುಷ',
        'phone_number': 'ದೂರವಾಣಿ ಸಂಖ್ಯೆ',
        'address': 'ವಿಳಾಸ',
        'farming_type': 'ಕೃಷಿ ಪ್ರಕಾರ',
        'register_farmer': 'ರೈತರನ್ನು ನೋಂದಾಯಿಸಿ',
        'farmer_registration': 'ರೈತ ನೋಂದಣಿ',
        'add': 'ಸೇರಿಸಿ',
        'delete_confirmation': 'ಡೇಟಾವನ್ನು ಅಳಿಸಲು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ?',
         'farming': 'ಕೃಷಿ',
         'rid': 'ಆರ್‌ಐಡಿ',
         'delete': 'ಅಳಿಸಿ',
          'add_agro_product': 'ಕೃಷಿ ಉತ್ಪನ್ನ ಸೇರಿಸಿ',
           'add_agro_products_title': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳನ್ನು ಸೇರಿಸಿ',
        'add_agro_products_heading': 'ಕೃಷಿ ಉತ್ಪನ್ನಗಳನ್ನು ಸೇರಿಸಿ',
        'farmer_name': 'ರೈತರ ಹೆಸರು',
        'farmer_email': 'ರೈತರ ಇಮೇಲ್',
        'farmer_email_placeholder': 'farmername@gmail.com',
        'product_name': 'ಉತ್ಪನ್ನದ ಹೆಸರು',
        'upload_product_image': 'ಉತ್ಪನ್ನದ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ',
        'choose_file': 'ಫೈಲ್ ಆಯ್ಕೆಮಾಡಿ',
        'no_file_chosen': 'ಯಾವುದೇ ಫೈಲ್ ಆಯ್ಕೆ ಮಾಡಲಾಗಿಲ್ಲ',
        'image_upload_optional': 'ಚಿತ್ರ ಫೈಲ್ ಆಯ್ಕೆಮಾಡಿ (JPG, PNG, ಇತ್ಯಾದಿ). ಐಚ್ಛಿಕ.',
        'product_description': 'ಉತ್ಪನ್ನ ವಿವರಣೆ',
        'price': 'ಬೆಲೆ',
        'unit': 'ಘಟಕ',
        'kg': 'ಕೆಜಿ',
        'quintal': 'ಕ್ವಿಂಟಲ್',
        'add_product_button': 'ಉತ್ಪನ್ನ ಸೇರಿಸಿ',
        'farmer_services': 'ರೈತ ಸೇವೆಗಳು',
        'resources': 'ಸಂಪನ್ಮೂಲಗಳು',
        'ai_tools': 'AI ಉಪಕರಣಗಳು',
          # Add Kannada translations for all keys
    }
}

##############################################################################################################
#Main Page 
@app.route('/')
def index():
    lang_code = request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
    return render_template('index.html', firestore=firestore, translations=selected_translations)

################################################################################################################


# Language chage function
def get_translations():
    lang_code = request.cookies.get('language', 'en')
    return TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])

############################################################################################################

#Register New Farmer (Link to Farming Types)
@app.route('/register', methods=['POST', 'GET'])
@login_required
def register():
    lang_code = request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])

    if db and current_user.is_authenticated:
        farming_ref = db.collection('farming_types')
        farming_types = farming_ref.get()
        farming_list = [{'id': doc.id, 'name': doc.to_dict().get('farmingtype')} for doc in farming_types]
        if request.method == "POST":
            farmername = request.form.get('farmername')
            adharnumber = request.form.get('adharnumber')
            age = request.form.get('age')
            gender = request.form.get('gender')
            phonenumber = request.form.get('phonenumber')
            address = request.form.get('address')
            farmingtype_id = request.form.get('farming_type')

            try:
                farming_type_doc = db.collection('farming_types').document(farmingtype_id).get()
                if farming_type_doc.exists:
                    farming_type_name = farming_type_doc.to_dict().get('farmingtype')
                    farmer_data = {
                        'user_id': current_user.id,
                        'farmername': farmername,
                        'adharnumber': adharnumber,
                        'age': int(age) if age else 0,
                        'gender': gender,
                        'phonenumber': phonenumber,
                        'address': address,
                        'farming': farming_type_name
                    }
                    db.collection('farmers').add(farmer_data)
                    return redirect('/farmerdetails')
                else:
                    flash(selected_translations['invalid_farming_type'], "danger") # Use translated message
                    return render_template('register.html', farming_types=farming_list, firestore=firestore, translations=selected_translations)
            except Exception as e:
                flash(selected_translations['registration_error'].format(e=e), "danger") # Use translated message
                return render_template('register.html', farming_types=farming_list, firestore=firestore, translations=selected_translations)
        else:
            return render_template('register.html', farming_types=farming_list, firestore=firestore, translations=selected_translations)
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

# Display Registered Farmer Details for Current User
@app.route('/farmerdetails')
@login_required
def farmerdetails():
    if db and current_user.is_authenticated:
        # Determine language and get translations FIRST, regardless of whether farmers are found
        lang= request.cookies.get('language', 'en') # default to English
        # Assuming TRANSLATIONS is a dictionary defined globally or imported
        # e.g., TRANSLATIONS = {'en': {...}, 'hi': {...}, ...}
        try:
            selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
        except NameError:
             # Handle case where TRANSLATIONS is not defined
             print("Error: TRANSLATIONS dictionary is not defined!")
             selected_translations = {} # Provide a default empty dictionary

        farmers_ref = db.collection('farmers')
        # Ensure 'user_id' field exists and is indexed in Firestore
        query = farmers_ref.where('user_id', '==', current_user.id).get()

        farmer_data = []
        # The loop now only processes the query results
        for farmer in query:
            farmer_dict = farmer.to_dict()
            farmer_dict['id'] = farmer.id # Add document ID to the dictionary
            farmer_data.append(farmer_dict)

        # selected_translations is now guaranteed to be defined here
        return render_template('farmerdetails.html', query=farmer_data, firestore=firestore, translations=selected_translations)
    else:
        # This block is for unauthenticated users
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))


# Delete a Registered Farmer Record with Authorization Check
@app.route("/delete/<string:farmer_id>", methods=['POST', 'GET'])
@login_required
def delete(farmer_id):
    if db and current_user.is_authenticated:
        try:
            farmer_ref = db.collection('farmers').document(farmer_id)
            farmer_doc = farmer_ref.get()
            if farmer_doc.exists:
                farmer_data = farmer_doc.to_dict()
                if 'user_id' in farmer_data and farmer_data['user_id'] == current_user.id:
                    farmer_ref.delete()
                    flash(f"Farmer with ID {farmer_id} deleted successfully.", "success")
                else:
                    flash("You are not authorized to delete this farmer.", "danger")
            else:
                flash(f"Farmer with ID {farmer_id} not found.", "warning")
        except Exception as e:
            flash(f"Error deleting farmer with ID {farmer_id}: {e}", "danger")
        return redirect('/farmerdetails')
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))


# API Endpoint for Client-Side Firebase User Registration & Firestore Data Storage
@app.route('/create_firebase_user', methods=['POST'])
def create_firebase_user():
    data = request.get_json()
    id_token = data.get('token')
    username = data.get('username')
    email = data.get('email')

    if not id_token:
        return jsonify({'success': False, 'error': 'No ID token provided'}), 400

    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token.get('uid')
        if uid:
            # Perform any server-side actions here, e.g.,
            # Create a user document in a 'farmers' collection
            user_data = {'uid': uid, 'username': username, 'email': email}
            db.collection('farmers').document(uid).set(user_data)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid ID token'}), 401
    except auth.InvalidIdTokenError:
        return jsonify({'success': False, 'error': 'Invalid ID token'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {e}'}), 500


# API Endpoint for Server-Side Firebase ID Token Verification & Flask-Login Authentication
@app.route('/verify_token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        id_token = data.get('token')

        if not id_token:
            return jsonify({'success': False, 'error': 'No token provided'})

        # Verify Firebase ID token
        decoded_token = firebase_auth.verify_id_token(id_token)

        uid = decoded_token['uid']
        email = decoded_token.get('email')

        # Flask-Login user creation
        user = User(uid)
        login_user(user)   # ⭐ THIS IS THE MOST IMPORTANT LINE

        print("LOGIN SUCCESS:", email)

        return jsonify({'success': True})

    except Exception as e:
        print("VERIFY TOKEN ERROR:", e)
        return jsonify({'success': False, 'error': str(e)})
    
#########################################################################################################################



###########################################################################################################

#Agro Products Management

UPLOAD_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Define allowed image file extensions

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add New Agro Product (with Image Upload and Validation)
@app.route('/addagroproducts', methods=['POST', 'GET'])
@login_required
def addagroproducts():
    lang_code = request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])

    if db and current_user.is_authenticated:
        if request.method == "POST":
            username = request.form.get('username')
            email = request.form.get('email')
            productname = request.form.get('productname')
            productdesc = request.form.get('productdesc')
            price = request.form.get('price')
            unit = request.form.get('unit') # Get the unit

            # --- Email Validation ---
            if not email or not email.endswith('@gmail.com'):
                flash("Please use a valid @gmail.com email address.", "danger")
                lang_code = request.cookies.get('language', 'en')
                selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
                # Optionally, re-render the template with form data to prevent loss
                return render_template('addagroproducts.html',
                                       firestore=firestore,
                                       translations=selected_translations,
                                       username=username,
                                       email=email,
                                       productname=productname,
                                       productdesc=productdesc,
                                       price=price,
                                       unit=unit) # Pass submitted data back

            # --- File Upload Handling ---
            file = request.files.get('product_image') # Get the file from the request

            image_filename = None # Initialize filename variable

            # Check if a file was uploaded and if it's allowed
            if file and file.filename != '' and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    image_filename = filename # Store only the filename
                    flash(f"Image '{filename}' uploaded successfully.", "success")
                except Exception as e:
                    flash(f"Error uploading image: {e}", "danger")
                    # Decide how to handle the error - maybe stop adding the product?
                    # For now, we'll add the product without the image if upload fails
                    image_filename = None # Ensure filename is None if upload fails
            elif file and file.filename != '' and not allowed_file(file.filename):
                flash("Invalid file type. Only PNG, JPG, JPEG, GIF are allowed.", "warning")
                lang_code = request.cookies.get('language', 'en')
                selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
                # Optionally, re-render the template with data
                return render_template('addagroproducts.html',
                                       firestore=firestore,
                                       translations=selected_translations,
                                       username=username,
                                       email=email,
                                       productname=productname,
                                       productdesc=productdesc,
                                       price=price,
                                       unit=unit) # Pass submitted data back
            
            # Prepare data for Firestore
            product_data = {
                'user_id': current_user.id, # Assuming current_user has an 'id' attribute
                'username': username,
                'email': email,
                'productname': productname,
                'productdesc': productdesc,
                'price': float(price) if price else 0.0, # Store price as float
                'unit': unit, # Store the unit (kg or quintal)
                'image_filename': image_filename # Store the filename or None
            }

            try:
                db.collection('agroproducts').add(product_data)
                flash("Product Added", "info")
                return redirect(url_for('agroproducts')) # Redirect after successful POST
            except Exception as e:
                flash(f"Error adding product to database: {e}", "danger")
                lang_code = request.cookies.get('language', 'en')
                selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
                # Optionally, re-render the template with data
                return render_template('addagroproducts.html',
                                       firestore=firestore,
                                       translations=selected_translations,
                                       username=username,
                                       email=email,
                                       productname=productname,
                                       productdesc=productdesc,
                                       price=price,
                                       unit=unit) # Pass submitted data back

        lang_code = request.cookies.get('language', 'en')
        selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
        # Render the form for GET requests
        return render_template('addagroproducts.html', firestore=firestore, translations=selected_translations)
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

# Delete Agro Product (with Image File Removal & Authorization)
@app.route("/deleteagroproduct/<string:product_id>", methods=['POST'])
@login_required
def deleteagroproduct(product_id):
    if db and current_user.is_authenticated:
        try:
            product_ref = db.collection('agroproducts').document(product_id)
            product_doc = product_ref.get()
            if product_doc.exists:
                product_data = product_doc.to_dict()
                # Check if the current user is the owner of the product
                if 'user_id' in product_data and product_data['user_id'] == current_user.id:
                    # --- NEW: Delete the image file if it exists ---
                    image_filename = product_data.get('image_filename') # Get the stored filename
                    if image_filename: # Check if a filename was stored
                         # Construct the full path to the image file
                         filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                         # Check if the file actually exists before trying to delete
                         if os.path.exists(filepath):
                             try:
                                 os.remove(filepath) # Delete the file
                                 # Optional: flash a message about file deletion
                                 # flash(f"Deleted image file: {image_filename}", "info")
                             except OSError as e:
                                 # Handle potential errors during file deletion (e.g., permissions)
                                 print(f"Error deleting image file {filepath}: {e}")
                                 flash(f"Could not delete image file: {image_filename}", "warning")
                         else:
                             # Optional: flash a message if the file was not found
                             # flash(f"Image file not found on server: {image_filename}", "warning")
                             pass # File not found, no action needed
                    # --- END NEW ---

                    # Delete the document from Firestore
                    product_ref.delete()
                    flash("Agro product deleted successfully.", "success")
                else:
                    flash("You are not authorized to delete this product.", "danger")
            else:
                flash("Agro product not found.", "warning")
        except Exception as e:
            # Catch any other exceptions during the process
            flash(f"Error deleting agro product: {e}", "danger")

        # Always redirect back to the agro products page after the operation
        return redirect(url_for('agroproducts')) # Using url_for is generally better

    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

#Display All Agro Products (for all users)
@app.route('/agroproducts')
def agroproducts():
    lang_code = request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])

    if db:
        try:
            agroproducts_ref = db.collection('agroproducts')
            agroproducts = agroproducts_ref.get()
            agroproducts_data = []
            for product in agroproducts:
                product_dict = product.to_dict()
                product_dict['id'] = product.id # Include document ID for delete function
                # Ensure 'user_id' exists, maybe add a default or handle missing field
                product_dict['user_id'] = product_dict.get('user_id') # Important for delete logic
                agroproducts_data.append(product_dict)

            return render_template('agroproducts.html',
                                   query=agroproducts_data,
                                   firestore=firestore,
                                   translations=selected_translations)
        except Exception as e:
            flash(f"Error fetching products: {e}", "danger")
            return render_template('agroproducts.html',
                                   query=[], # Pass empty list on error
                                   firestore=firestore,
                                   translations=selected_translations)
    else:
        flash("Database connection error.", "danger")
        return render_template('agroproducts.html',
                               query=[], # Pass empty list on error
                               firestore=firestore,
                               translations=selected_translations)

# Add New Farming Type to Database
@app.route('/addfarming', methods=['POST', 'GET'])
@login_required
def addfarming():
    if db and current_user.is_authenticated:
        if request.method == "POST":
            farmingtype = request.form.get('farming')
            farming_ref = db.collection('farming_types').where('farmingtype', '==', farmingtype).limit(1).get()
            if farming_ref:
                for doc in farming_ref:
                    flash("Farming Type Already Exist", "warning")
                    return redirect('/addfarming')
            farming_data = {'farmingtype': farmingtype}
            db.collection('farming_types').add(farming_data)
            flash("Farming Added", "success")
        lang= request.cookies.get('language', 'en') # default to English
        selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
        return render_template('farming.html', firestore=firestore,translations=selected_translations)
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

######################################################################################################################

#function is intended to handle the editing of a record identified by rid 
@app.route("/edit/<string:rid>", methods=['POST', 'GET'])
@login_required
def edit(rid):
    flash("Edit logic needs to be updated for Firestore.", "warning")
    return render_template('edit.html', posts=None, farming=None, firestore=firestore) # Placeholder

############################################################################################################

#################################################################################################################
#User Account Registration with Firebase Authentication and Firestore
@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if firebase_auth:
        if request.method == "POST":
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            try:
                user = firebase_auth.create_user(email=email, password=password, display_name=username)
                # Create a corresponding document in the 'users' collection
                user_data = {'username': username, 'email': email}
                db.collection('users').document(user.uid).set(user_data)
                flash("Signup Successful! Please Login.", "success")
                return redirect(url_for('login'))
            except firebase_admin.auth.EmailAlreadyExistsError:
                flash("Email Already Exists", "warning")
            except Exception as e:
                flash(f"Signup Failed: {e}", "danger")
            return render_template('signup.html', firestore=firestore)
        return render_template('signup.html', firestore=firestore)
    else:
        flash("Firebase Auth not initialized.", "danger")
        return render_template('signup.html', firestore=firestore)

#User Login Form Display
@app.route('/login', methods=['POST', 'GET'])
def login():
    return render_template('login.html', firestore=firestore)

# User Session Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logout Successful", "warning")
    return redirect(url_for('login'))

#######################################################################################################################


##############################################################################################################

#Test Firestore Database Connectivity@app.route('/test')
def test():
    if db:
        try:
            docs = db.collection('agroproducts').get()
            return f'Firestore is connected. Found {len(docs)} documents in agroproducts.'
        except Exception as e:
            return f'Error connecting to Firestore: {e}'
    else:
        return 'Firestore not initialized.'

################################################################################################################
# Crop Recommendation Route
@app.route('/preone')
def preone():
    return render_template("preone.html", firestore=firestore)

#Perform Crop Recommendation Prediction
@app.route("/predict", methods=['POST'])
def predict():
    if model is not None and sc is not None and mx is not None:
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosporus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)
            mx_features = mx.transform(single_pred)
            sc_mx_features = sc.transform(mx_features)
            prediction = model1.predict(sc_mx_features)
            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = f"{crop} is the best crop to be cultivated right there"
            else:
                result = "Sorry, we could not determine the best crop."
            return render_template('preone.html', result=result, firestore=firestore)
        except (ValueError, KeyError):
            return render_template('preone.html', result="Invalid input.", firestore=firestore)
        except Exception as e:
            return render_template('preone.html', result=f"An unexpected error occurred: {e}", firestore=firestore)
    else:
        return render_template('preone.html', result="Crop prediction models are not loaded.", firestore=firestore)


#####################################################################################################################

#STATIC PAGES OF THE WEBSITE
@app.route('/govschems')
def govschems():
    lang= request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    return render_template('govschems.html', firestore=firestore,translations=selected_translations)
@app.route('/Plantmood')
def Plantmood():
    lang= request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    return render_template('Plantmood.html', firestore=firestore,translations=selected_translations)

@app.route('/map')
def map():
    return render_template('map.html', firestore=firestore)

@app.route('/transport')
def transport():
    lang= request.cookies.get('language', 'en')
    selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    return render_template('transport.html', firestore=firestore,translations=selected_translations)



###################################################################################################################

##################################################################################################################
#Ai crop roadmap providing
@app.route('/croproadmap')
def croproadmap():
    return render_template('croproadmap.html', firestore=firestore)

##########################################################################################################################
#Display Agricultural Equipment with Filters
@app.route('/equipment')
def equipment():
    if db:
        equipment_list = db.collection('equipment').get()
        categories = set()
        brands = set()
        equipment_data = []
        for equip in equipment_list:
            data = equip.to_dict()
            data['id'] = equip.id
            equipment_data.append(data)
            if 'category' in data:
                categories.add(data['category'])
            if 'brand' in data:
                brands.add(data['brand'])

        category_filter = request.args.get('category')
        brand_filter = request.args.get('brand')

        filtered_equipment = []
        for equip in equipment_data:
            include = True
            if category_filter and equip.get('category') != category_filter:
                include = False
            if brand_filter and equip.get('brand') != brand_filter:
                include = False
            if include:
                filtered_equipment.append(equip)
        lang= request.cookies.get('language', 'en')  # default to English
        selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
        return render_template('equipment.html', equipment=filtered_equipment, categories=categories, brands=brands, current_category=category_filter, current_brand=brand_filter, firestore=firestore,translations=selected_translations)
    else:
        flash("Database connection error.", "danger")
        lang= request.cookies.get('language', 'en')  # default to English
        selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
        return render_template('equipment.html', equipment=[], categories=[], brands=[], current_category=None, current_brand=None, firestore=firestore,translations=selected_translations)
    
#Add Equipment to User's Shopping Cart
@app.route('/equipment/add_to_cart', methods=['POST'])
@login_required
def add_to_cart():
    if db and current_user.is_authenticated:
        equipment_id = request.form['equipment_id']
        print(f"ADD TO CART USER ID: {current_user.id}")
        print(f"Attempting to add equipment with ID: {equipment_id}")
        equipment_ref = db.collection('equipment').document(equipment_id)
        equipment_doc = equipment_ref.get()

        if equipment_doc.exists:
            equipment_data = equipment_doc.to_dict()
            price = equipment_data.get('price')
            print(f"Equipment data: {equipment_data}")

            if price is None:
                flash(f"{equipment_data.get('name')} has no price set...", "warning")
                return redirect('/equipment')
            if not isinstance(price, (int, float)):
                flash(f"{equipment_data.get('name')} has an invalid price...", "warning")
                return redirect('/equipment')

            # Convert available_quantity to an integer before comparison
            try:
                available_quantity = int(equipment_data.get('available_quantity', 0))
            except ValueError:
                flash(f"{equipment_data.get('name')} has an invalid available quantity.", "warning")
                return redirect('/equipment')

            if available_quantity > 0:
                user_cart_ref = db.collection('users').document(current_user.id).collection('cart')
                cart_item_query = user_cart_ref.where('equipment_id', '==', equipment_id).limit(1).get()
                cart_item = [item for item in cart_item_query]

                if cart_item:
                    print(f"Item already in cart: {cart_item[0].to_dict()}")
                    cart_item_ref = user_cart_ref.document(cart_item[0].id)
                    cart_item_data = cart_item[0].to_dict()
                    new_quantity = cart_item_data.get('quantity', 0) + 1
                    
                    # Ensure comparison with integer available_quantity
                    if new_quantity <= available_quantity:
                        update_data = {'quantity': new_quantity, 'price': price}
                        cart_item_ref.update(update_data)
                        equipment_ref_update = db.collection('equipment').document(equipment_id)
                        equipment_ref_update.update({'available_quantity': available_quantity - 1})
                        print(f"Updated cart item quantity to: {new_quantity}")
                        flash(f"Added one {equipment_data.get('name')} to cart.", "success")
                    else:
                        flash(f"Not enough {equipment_data.get('name')} in stock.", "warning")
                else:
                    new_cart_item = {
                        'equipment_id': equipment_id,
                        'name': equipment_data.get('name'),
                        'price': price,
                        'quantity': 1
                    }
                    print(f"Adding new item to cart: {new_cart_item}")
                    user_cart_ref.add(new_cart_item)
                    equipment_ref_update = db.collection('equipment').document(equipment_id)
                    equipment_ref_update.update({'available_quantity': available_quantity - 1})
                    flash(f"Added {equipment_data.get('name')} to cart.", "success")
            else:
                flash(f"{equipment_data.get('name')} is out of stock.", "warning")
        else:
            flash("Equipment not found.", "danger")
        return redirect('/equipment')
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

# Display User's Shopping Cart Contents , user specific
@app.route('/cart')
@login_required
def cart():
    if db and current_user.is_authenticated:
        user_id = current_user.id
        user_cart_ref = db.collection('users').document(user_id).collection('cart')
        try:
            cart_items = user_cart_ref.get()
            cart_data = []
            total_amount = 0  # Initialize total

            for item in cart_items:
                cart_item = item.to_dict()
                cart_item['id'] = item.id
                cart_data.append(cart_item)
                # Add to total
                total_amount += cart_item.get('price', 0) * cart_item.get('quantity', 0)
            lang= request.cookies.get('language', 'en')  # default to English
            selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
            return render_template('cart.html', cart_items=cart_data, total_amount=total_amount, firestore=firestore,translations=selected_translations)
        except Exception as e:
            flash("Error loading cart.", "danger")
            lang= request.cookies.get('language', 'en')  # default to English
            selected_translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
            return render_template('cart.html', cart_items=[], total_amount=0, firestore=firestore,translations=selected_translations)
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

#Remove an Item from User's Shopping Cart
@app.route('/cart/remove/<item_id>', methods=['POST'])
@login_required
def remove_from_cart(item_id):
    if db and current_user.is_authenticated:
        user_cart_ref = db.collection('users').document(current_user.id).collection('cart')
        item_ref = user_cart_ref.document(item_id)
        item_ref.delete()
        flash("Item removed from cart.", "info")
        return redirect('/cart')
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))

#Update Quantity of an Item in User's Shopping Cart
@app.route('/cart/update/<item_id>', methods=['POST'])
@login_required
def update_cart_quantity(item_id):
    if db and current_user.is_authenticated:
        try:
            new_quantity = int(request.form['quantity'])
            if new_quantity < 1:
                flash("Quantity must be at least 1.", "warning")
                return redirect('/cart')

            user_cart_ref = db.collection('users').document(current_user.id).collection('cart')
            item_ref = user_cart_ref.document(item_id)
            item_doc = item_ref.get()

            if not item_doc.exists:
                flash("Cart item not found.", "danger")
                return redirect('/cart')

            item_data = item_doc.to_dict()
            old_quantity = item_data.get('quantity', 0)
            quantity_diff = new_quantity - old_quantity

            equipment_id = item_data.get('equipment_id')
            equipment_ref = db.collection('equipment').document(equipment_id)
            equipment_doc = equipment_ref.get()

            if equipment_doc.exists:
                equipment_data = equipment_doc.to_dict()
                current_stock = equipment_data.get('available_quantity', 0)

                if quantity_diff > 0 and current_stock < quantity_diff:
                    flash("Not enough stock available to increase quantity.", "warning")
                    return redirect('/cart')

                # Update available_quantity accordingly
                new_stock = current_stock - quantity_diff
                equipment_ref.update({'available_quantity': new_stock})

                # Update the cart quantity
                item_ref.update({'quantity': new_quantity})
                flash("Cart updated.", "success")
            else:
                flash("Equipment not found.", "danger")
        except Exception as e:
            print(f"Update error: {e}")
            flash("Error updating cart.", "danger")

        return redirect('/cart')
    else:
        flash("Authentication required.", "danger")
        return redirect(url_for('login'))
############################################################################################################################

# Configuration for file uploads
# Construct the path relative to the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_logged_in_user():
    if 'user_id' in session:
        # In a real app, you might fetch user details from Firestore
        # based on session['user_id']
        return {'uid': session['user_id'], 'role': session.get('user_role', 'farmer')}
    return None

#############################################################################################################

#Check if the current user is a logged-in worker
def is_worker_logged_in():
    user = get_logged_in_user()
    return user and user['role'] == 'worker'

#Display All Registered Workers
@app.route('/worker')
def worker_home():
    workers_ref = db.collection('workers')
    workers_data = []
    try:
        for doc in workers_ref.stream():
            worker = doc.to_dict()
            worker['id'] = doc.id # Good practice
            workers_data.append(worker)
    except Exception as e:
        flash(f"Error fetching workers: {str(e)}", "error")
        print(f"Error in worker_home: {e}") # For server-side logging
    return render_template('worker.html', workers=workers_data, current_user=get_logged_in_user())

#Display Worker signup Form
@app.route('/signup1.html')
def worker_signup():
    return render_template('signup1.html')

#Process Worker Account Registration
@app.route('/worker/signup', methods=['POST'])
def worker_signup_post():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if not all([username, email, password, confirm_password]):
        flash('All fields are required.', 'error')
        return render_template('signup1.html', username=username, email=email)

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return render_template('signup1.html', error='Passwords do not match.', username=username, email=email)

    users_ref = db.collection('users')
    username_query = users_ref.where('username', '==', username).limit(1).stream()
    email_query = users_ref.where('email', '==', email).limit(1).stream()

    if len(list(username_query)) > 0:
        flash('Username already taken.', 'error')
        return render_template('signup1.html', error='Username already taken.', username=username, email=email)
    if len(list(email_query)) > 0:
        flash('Email already registered.', 'error')
        return render_template('signup1.html', error='Email already registered.', username=username, email=email)

    # IMPORTANT: Hash passwords in a real application!
    # from werkzeug.security import generate_password_hash
    # hashed_password = generate_password_hash(password)
    user_data = {'username': username, 'email': email, 'password': password, 'role': 'worker'}
    
    try:
        new_user_ref = users_ref.document()
        new_user_ref.set(user_data)
        session['user_id'] = new_user_ref.id
        session['user_role'] = 'worker'
        flash('Signup successful! Please complete your worker profile.', 'success')
        return redirect(url_for('worker_dashboard'))
    except Exception as e:
        flash(f'An error occurred during signup: {str(e)}', 'error')
        print(f"Error in worker_signup_post: {e}")
        return render_template('signup1.html', username=username, email=email)

# Display Worker Login Form
@app.route('/login1.html')
def worker_login():
    return render_template('login1.html')

#Process Worker Account Login
@app.route('/worker/login', methods=['POST'])
def worker_login_post():
    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        flash('Username and password are required.', 'error')
        return render_template('login1.html', username=username)

    users_ref = db.collection('users')
    # IMPORTANT: Compare hashed passwords in a real application!
    # from werkzeug.security import check_password_hash
    query_stream = users_ref.where('username', '==', username).where('password', '==', password).limit(1).stream()
    user_list = list(query_stream)

    if user_list:
        user_doc = user_list[0]
        user_data = user_doc.to_dict()
        session['user_id'] = user_doc.id
        session['user_role'] = user_data.get('role', 'farmer')
        
        if session['user_role'] == 'worker':
            flash('Login successful!', 'success')
            return redirect(url_for('worker_dashboard'))
        else:
            session.pop('user_id', None)
            session.pop('user_role', None)
            flash('These credentials are not for a worker account.', 'error')
            return render_template('login1.html', error='Invalid worker credentials.', username=username)
    else:
        flash('Invalid username or password.', 'error')
        return render_template('login1.html', error='Invalid username or password.', username=username)

# Worker's Personal Dashboard / Profile Management
@app.route('/worker/dashboard')
def worker_dashboard():
    if not is_worker_logged_in():
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('worker_login'))

    user_id = session['user_id']
    workers_ref = db.collection('workers')
    worker_query_stream = workers_ref.where('userId', '==', user_id).limit(1).stream()
    
    worker_data = None
    worker_list = list(worker_query_stream)
    if worker_list:
        worker_data = worker_list[0].to_dict()
        worker_data['doc_id'] = worker_list[0].id # Store doc_id for easy updates

    # This template should contain the form for registering/updating worker profile
    return render_template('worker1.html', worker_data=worker_data, current_user=get_logged_in_user())


## Action: Register/Update Worker Profile Details (Initial or Subsequent)
@app.route('/worker/register', methods=['POST'])
def register_worker():
    if not is_worker_logged_in():
        return redirect(url_for('worker_login'))

    user_id = session['user_id']
    name = request.form.get('name')
    image_url = request.form.get('image_url')
    experience = request.form.get('experience')
    work_details = request.form.get('work_details')
    per_day_price = request.form.get('per_day_price')
    expertise = request.form.get('expertise')
    contact_number = request.form.get('contact_number')

    worker_data = {
        'userId': user_id,
        'name': name,
        'imageUrl': image_url,
        'experience': experience,
        'workDetails': work_details,
        'perDayPrice': float(per_day_price) if per_day_price else 0.0,
        'expertise': expertise,
        'availability': 'Available',
        'contactNumber': int(contact_number) if contact_number else 0 # Assuming contact number is an integer
    }

    workers_ref = db.collection('workers')
    worker_query = workers_ref.where('userId', '==', user_id).limit(1).get()

    if list(worker_query):
        # Worker already registered, update their profile
        worker_doc_id = list(worker_query)[0].id
        workers_ref.document(worker_doc_id).update(worker_data)
    else:
        # Register the worker for the first time
        workers_ref.add(worker_data)

    return redirect(url_for('worker_dashboard'))


##Update Worker Profile Details
@app.route('/worker/update', methods=['POST'])
def update_worker_profile():
    if not is_worker_logged_in():
        return redirect(url_for('worker_login'))

    user_id = session['user_id']
    name = request.form.get('name')
    image_url = request.form.get('image_url')
    experience = request.form.get('experience')
    work_details = request.form.get('work_details')
    per_day_price = request.form.get('per_day_price')
    expertise = request.form.get('expertise')
    availability = request.form.get('availability')
    contact_number = request.form.get('contact_number')

    worker_data = {
        'name': name,
        'imageUrl': image_url,
        'experience': experience,
        'workDetails': work_details,
        'perDayPrice': float(per_day_price) if per_day_price else 0.0,
        'expertise': expertise,
        'availability': availability,
        'contactNumber': int(contact_number) if contact_number else 0 # Assuming contact number is an integer
    }

    workers_ref = db.collection('workers')
    worker_query = workers_ref.where('userId', '==', user_id).limit(1).get()

    if list(worker_query):
        worker_doc_id = list(worker_query)[0].id
        workers_ref.document(worker_doc_id).update(worker_data)
    return redirect(url_for('worker_dashboard'))



# Save/Update Worker Profile (Comprehensive Validation & Image Upload)
@app.route('/worker/profile/save', methods=['POST'])
def save_worker_profile():
    if not is_worker_logged_in():
        flash('Authentication required to save profile.', 'error')
        return redirect(url_for('worker_login'))

    user_id = session['user_id']

    name = request.form.get('name')
    experience = request.form.get('experience')
    work_details = request.form.get('work_details')
    per_day_price_str = request.form.get('per_day_price')
    expertise = request.form.get('expertise')
    contact_number_str = request.form.get('contact_number', "").strip() # Get and strip whitespace
    availability = request.form.get('availability', 'Available')

    # --- Server-Side Validation for Contact Number ---
    if not re.match(r"^\d{10}$", contact_number_str): # Check for exactly 10 digits
        flash("Contact number must be exactly 10 digits.", "error")
        # To retain form data on error, prepare it for the template
        current_form_data = request.form.to_dict()
        # If updating, try to fetch existing worker data to merge
        workers_ref_temp = db.collection('workers')
        worker_query_temp = workers_ref_temp.where('userId', '==', user_id).limit(1).stream()
        worker_data_temp_list = list(worker_query_temp)
        worker_data_for_template = worker_data_temp_list[0].to_dict() if worker_data_temp_list else {}
        worker_data_for_template.update(current_form_data) # Overwrite with current (potentially invalid) form data to show back to user
        if 'imageUrl' not in worker_data_for_template and worker_data_temp_list and 'imageUrl' in worker_data_temp_list[0].to_dict():
             worker_data_for_template['imageUrl'] = worker_data_temp_list[0].to_dict()['imageUrl'] # Preserve image if not re-uploaded

        return render_template('worker1.html', worker_data=worker_data_for_template, current_user=get_logged_in_user())
    # --- End Contact Number Validation ---

    # Basic validation for other fields (ensure this is comprehensive)
    if not all([name, experience, per_day_price_str, expertise]): # contact_number_str already validated for format
        flash("Please fill in all required fields (Name, Experience, Price, Expertise, Contact).", "error")
        current_form_data = request.form.to_dict()
        workers_ref_temp = db.collection('workers')
        worker_query_temp = workers_ref_temp.where('userId', '==', user_id).limit(1).stream()
        worker_data_temp_list = list(worker_query_temp)
        worker_data_for_template = worker_data_temp_list[0].to_dict() if worker_data_temp_list else {}
        worker_data_for_template.update(current_form_data)
        if 'imageUrl' not in worker_data_for_template and worker_data_temp_list and 'imageUrl' in worker_data_temp_list[0].to_dict():
             worker_data_for_template['imageUrl'] = worker_data_temp_list[0].to_dict()['imageUrl']

        return render_template('worker1.html', worker_data=worker_data_for_template, current_user=get_logged_in_user())


    image_db_filename = None

    if 'profile_image' in request.files:
        file = request.files['profile_image']
        if file.filename == '':
            pass
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_db_filename = filename
            except Exception as e:
                flash(f"Could not save image: {str(e)}", "error")
                print(f"Error saving image: {e}")
                # Re-render form with existing data
                workers_ref_temp = db.collection('workers')
                worker_query_temp = workers_ref_temp.where('userId', '==', user_id).limit(1).stream()
                worker_data_temp_list = list(worker_query_temp)
                worker_data_for_template = worker_data_temp_list[0].to_dict() if worker_data_temp_list else {}
                worker_data_for_template.update(request.form.to_dict()) # keep other form inputs
                return render_template('worker1.html', worker_data=worker_data_for_template, current_user=get_logged_in_user())

        elif file.filename != '':
            flash("Invalid image file type. Allowed types: png, jpg, jpeg, gif.", "error")
            workers_ref_temp = db.collection('workers')
            worker_query_temp = workers_ref_temp.where('userId', '==', user_id).limit(1).stream()
            worker_data_temp_list = list(worker_query_temp)
            worker_data_for_template = worker_data_temp_list[0].to_dict() if worker_data_temp_list else {}
            worker_data_for_template.update(request.form.to_dict())
            return render_template('worker1.html', worker_data=worker_data_for_template, current_user=get_logged_in_user())

    worker_profile_data = {
        'userId': user_id,
        'name': name,
        'experience': experience,
        'workDetails': work_details,
        'perDayPrice': float(per_day_price_str) if per_day_price_str else 0.0,
        'expertise': expertise,
        'availability': availability,
        'contactNumber': contact_number_str # Store the validated 10-digit number
    }

    workers_ref = db.collection('workers')
    worker_query_stream = workers_ref.where('userId', '==', user_id).limit(1).stream()
    existing_worker_list = list(worker_query_stream)

    try:
        if existing_worker_list:
            worker_doc_ref = workers_ref.document(existing_worker_list[0].id)
            existing_data = existing_worker_list[0].to_dict()
            if image_db_filename:
                worker_profile_data['imageUrl'] = image_db_filename
            elif 'imageUrl' in existing_data:
                worker_profile_data['imageUrl'] = existing_data['imageUrl']
            else: # No new image and no existing image
                 worker_profile_data['imageUrl'] = None


            worker_doc_ref.update(worker_profile_data)
            flash('Profile updated successfully!', 'success')
        else:
            if image_db_filename:
                worker_profile_data['imageUrl'] = image_db_filename
            else:
                worker_profile_data['imageUrl'] = None
            workers_ref.add(worker_profile_data)
            flash('Profile registered successfully!', 'success')
    except Exception as e:
        flash(f"Error saving profile to database: {str(e)}", "error")
        print(f"Firestore error in save_worker_profile: {e}")

    return redirect(url_for('worker_dashboard'))


#Worker Session Logout
@app.route('/worker/logout')
def worker_logout():
    session.pop('user_id', None)
    session.pop('user_role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('worker_home')) # Or your main landing page


if __name__ == '__main__':
    app.run(debug=True)