from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from azureml.core import Workspace
from train import train_iris_model, train_baseline_model
from azureml.core.authentication import ServicePrincipalAuthentication
import os
from pathlib import Path

def train_and_register():
    try:
        # Service Principal Authentication - REMOVE CREDENTIALS FROM CODE!
        # Use environment variables or Azure Key Vault instead
        svc_pr = ServicePrincipalAuthentication(
            tenant_id="acb2f2f3-9488-4fec-95cc",
            service_principal_id="0cda455c-bf57-44a1-ac02",
            service_principal_password="yunErUbMEueZrrQNV4mgoyMf231Q83-by1"
        )
        # Get absolute path to config.json relative to DAG file
        dag_dir = Path(__file__).parent
        config_path = dag_dir.parent / "config" / "config.json"
        
        print(f"Loading config from: {config_path}")
        
        # Get workspace using Service Principal
        ws = Workspace.from_config(str(config_path), auth=svc_pr)

        # Get workspace using Service Principal
        #ws = Workspace.from_config('./config/config.json', auth=svc_pr)
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        
        # Set the experiment name to see it in Azure ML Studio
        mlflow.set_experiment("iris-classification-hpo")
        
        with mlflow.start_run(run_name="iris_hpo_training") as run:
            print(f"Starting MLflow run: {run.info.run_id}")
            
            # Train the model with hyperparameter optimization
            model = train_iris_model()
            
            # Get metrics from the current run
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run.info.run_id)
            final_accuracy = run_info.data.metrics.get('final_accuracy', 0)
            
            print(f"Final accuracy: {final_accuracy}")
            
            # Only register if accuracy meets threshold
            if final_accuracy > 0.9:
                model_uri = f"runs:/{run.info.run_id}/iris_model"
                registered_model = mlflow.register_model(model_uri, "iris-classifier")
                
                # Add model description with performance metrics
                client.update_registered_model(
                    name="iris-classifier",
                    description=f"Iris classifier with accuracy: {final_accuracy:.4f}"
                )
                
                print(f"✅ Model registered with accuracy: {final_accuracy:.4f}")
                print(f"✅ Model version: {registered_model.version}")
            else:
                print(f"❌ Model accuracy {final_accuracy:.4f} below threshold (0.9), training baseline...")
                # Fallback to baseline model
                baseline_model = train_baseline_model()
                mlflow.sklearn.log_model(baseline_model, "baseline_model")
                print("✅ Baseline model trained as fallback")
                
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Define the DAG - FIXED schedule_interval parameter
dag = DAG(
    'iris_retrain_hpo',
    default_args={
        'start_date': datetime(2025, 8, 24),
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    schedule='@weekly',  # Fixed parameter name
    catchup=False,
    max_active_runs=1
)

task = PythonOperator(
    task_id='retrain_with_hpo',
    python_callable=train_and_register,
    dag=dag,
    execution_timeout=timedelta(hours=2)
)