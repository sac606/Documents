import streamlit as st
import numpy as np
import mlflow
import mlflow.sklearn
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
import joblib
import os
#from dotenv import load_dotenv

# Load environment variables from .env file
#load_dotenv()

# Service Principal Authentication
svc_pr = ServicePrincipalAuthentication(
    tenant_id="acb2f2f3-9488-4fec-95cc",
    service_principal_id="0cda455c-bf57-44a1",
    service_principal_password="yunErUbMEueZrrQNV4mgoyMf231Q83-by1"
)

# Get workspace using Service Principal
ws = Workspace.from_config('./config.json', auth=svc_pr)
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# Load model using Azure ML SDK instead of MLflow registry
@st.cache_resource
def load_model():
    try:
        # Method 1: Load using Azure ML SDK (more reliable)
        model = Model(ws, name="iris-classifier", version=2)  # Use your specific version   
        # best wat to create endpoint for model and directly call endpoint
        model_path = model.download(exist_ok=True)
        print(model_path)
        # Load the model file directly
        model_file = os.path.join(model_path)#, "model.pkl")  # Adjust if different filename
        print(model_file)
        return joblib.load(model_file)
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("Iris Classifier")

# Input sliders for Iris features
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3)

if st.button("Predict"):
    if model is not None:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        try:
            prediction = model.predict(input_data)
            # Map numeric predictions to class names if needed
            class_names = ['setosa', 'versicolor', 'virginica']
            predicted_class = class_names[prediction[0]] if isinstance(prediction[0], (int, np.integer)) else prediction[0]
            st.success(f"Predicted class: {predicted_class}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else
        st.error("Model not loaded. Please check authentication and model availability.")