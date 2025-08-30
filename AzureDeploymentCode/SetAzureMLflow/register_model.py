import mlflow
import mlflow.sklearn
from azureml.core import Workspace, Model
import joblib
from azureml.core.authentication import AzureCliAuthentication
import os

# Use Azure CLI authentication
cli_auth = AzureCliAuthentication()
ws = Workspace.from_config('./config.json', auth=cli_auth)

# Set up MLflow tracking (for experiment tracking only)
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("iris-experiment")

with mlflow.start_run():
    # Load model
    model = joblib.load("model.pkl")
    
    # Log parameters and metrics for tracking
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("registration_method", "AzureML_SDK")
    
    # Save model temporarily
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    
    # Register model using Azure ML SDK (reliable method)
    registered_model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name="iris-classifier",
        description="Random Forest classifier for iris dataset"
    )
    
    # Log registration info
    mlflow.log_param("registered_model_name", registered_model.name)
    mlflow.log_param("model_version", registered_model.version)
    
    print(f"✅ Model registered: {registered_model.name}, version: {registered_model.version}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)