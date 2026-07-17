"""
Define a sweep config, initialize, and start W&B Sweep.

sweep_configuration is a single nested sweep configuration. See sweep_config.py
for an example of a double nested sweep configuration.
"""
import wandb
import numpy as np
import random
import argparse

def train_one_epoch(epoch, lr, batch_size):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

def main(args=None):
    # When called by sweep agent, args will be None,
    # so we use the project from sweep config
    project = args.project if args else None
    
    with wandb.init(project=project) as run:
        # Fetches the hyperparameter values from `wandb.Run.config` object
        lr = run.config["lr"]
        batch_size = run.config["batch_size"]
        epochs = run.config["epochs"]

        # Execute the training loop and log the performance values to W&B
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
            val_acc, val_loss = evaluate_one_epoch(epoch)
            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc, # Metric optimized
                    "val_loss": val_loss,
                }
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="sweep-example", help="W&B project name")
    args = parser.parse_args()

    # Define a sweep config dictionary
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {
            "goal": "maximize",
            "name": "val_acc"
            },
        "parameters": {
            "batch_size": {"values": [16, 32, 64]},
            "epochs": {"values": [5, 10, 15]},
            "lr": {"max": 0.1, "min": 0.0001},
        },
    }

    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

    # Start the sweep job
    wandb.agent(sweep_id, function=main, count=2)