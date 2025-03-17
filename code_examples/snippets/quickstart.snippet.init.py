run = wandb.init(
    project="my-awesome-project",    # Specify your project
    config={                         # Track hyperparameters and metadata
        "learning_rate": lr,
        "epochs": epochs,
    },
)
