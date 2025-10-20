
def model(training_data: int) -> int:
    """Model simulation for demonstration purposes."""
    return training_data * 2 + random.randint(-1, 1)

# Simulate weights and noise
weights = random.random() # Initialize random weights
noise = random.random() / 5  # Small random noise to simulate noise

# Hyperparameters and configuration
config = {
    "epochs": 10,  # Number of epochs to train
    "learning_rate": 0.01,  # Learning rate for the optimizer
}

# Use context manager to initialize and close W&B runs
with wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config) as run:
    # Simulate training loop
    for epoch in range(config["epochs"]):
        xb = weights + noise  # Simulated input training data
        yb = weights + noise * 2  # Simulated target output (double the input noise)

        y_pred = model(xb)  # Model prediction
        loss = (yb - y_pred) ** 2  # Mean Squared Error loss

        print(f"epoch={epoch}, loss={y_pred}")
        # Log epoch and loss to W&B
        run.log({
            "epoch": epoch,
            "loss": loss,
        })

    # Unique name for the model artifact,
    model_artifact_name = f"model-demo"

    # Local path to save the simulated model file
    PATH = "model.txt"

    # Save model locally
    with open(PATH, "w") as f:
        f.write(str(weights)) # Saving model weights to a file

    # Create an artifact object
    # Add locally saved model to artifact object
    artifact = wandb.Artifact(name=model_artifact_name, type="model", description="My trained model")
    artifact.add_file(local_path=PATH)
    artifact.save()

