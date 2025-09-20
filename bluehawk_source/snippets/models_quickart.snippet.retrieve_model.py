
REGISTRY_NAME = "Model"  # Name of the registry in W&B
COLLECTION_NAME = "DemoModels"  # Name of the collection in the registry
VERSION = 0 # Version of the artifact to retrieve

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()

