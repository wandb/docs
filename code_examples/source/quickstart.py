#!/usr/bin/env python3

try:
    # :snippet-start: all
    # :snippet-start: import
    import wandb
    import random
    # :snippet-end: import

    # :snippet-start: login
    wandb.login()
    # :snippet-end: login

    epochs = 10
    lr = 0.01

    # :snippet-start: init
    run = wandb.init(
        project="my-awesome-project",    # Specify your project
        config={                         # Track hyperparameters and metadata
            "learning_rate": lr,
            "epochs": epochs,
        },
    )
    # :snippet-end: init

    offset = random.random() / 5
    print(f"lr: {lr}")

    # Simulate a training run
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        wandb.log({"accuracy": acc, "loss": loss})
    run.finish()
    # :snippet-end: all
    exit(0)
except Exception as e:
    print(f"Error: {e}")
    exit(1)