# src/model/promote_model.py
import mlflow
from mlflow.tracking import MlflowClient
# Initialize DagsHub + MLflow
import dagshub
dagshub.init(repo_owner="emaljm", repo_name="MLops-new", mlflow=True)


MODEL_NAME = "SentimentClassifier"

def promote_best_model(model_name: str = MODEL_NAME):
    client = MlflowClient()
    
    # Get all model versions
    versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
    
    if not versions:
        print("No registered model versions found.")
        return
    
    # Select the best version based on metrics (e.g., F1 score)
    best_version = None
    best_f1 = -1
    for v in versions:
        run = client.get_run(v.run_id)
        metrics = run.data.metrics
        f1 = metrics.get("f1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_version = v.version
    
    if best_version is None:
        print("No suitable model found to promote.")
        return
    
    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Promoted model version {best_version} to Production with F1: {best_f1}")


if __name__ == "__main__":
    promote_best_model()
