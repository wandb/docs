
# Artifact name specifies the specific artifact version within our team's project
artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
print("Artifact name: ", artifact_name)

REGISTRY_NAME = "Model" # Name of the registry in W&B
COLLECTION_NAME = "DemoModels"  # Name of the collection in the registry

# Create a target path for our artifact in the registry
target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
print("Target path: ", target_path)

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
model_artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
run.link_artifact(artifact=model_artifact, target_path=target_path)
run.finish()

