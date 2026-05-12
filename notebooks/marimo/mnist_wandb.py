# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wandb>=0.17",
#     "torch>=2.0",
#     "torchvision>=0.15",
# ]
# ///
"""MNIST handwritten-digit classifier with Weights & Biases tracking.

Run interactively:
    uvx marimo edit --sandbox notebooks/marimo/mnist_wandb.py

Run as a script:
    uvx --with marimo --with wandb --with torch --with torchvision python \\
        notebooks/marimo/mnist_wandb.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(
        """
        # Train an MNIST classifier with W&B Models

        This notebook trains a small convolutional network on the MNIST
        handwritten-digit dataset and tracks the run with
        [Weights & Biases Models](https://docs.wandb.ai/models/).

        Adjust the hyperparameters below, then press **Start training run**.
        Each press creates a new W&B run that records:

        - per-batch training loss and per-epoch validation accuracy,
        - parameter and gradient histograms (via `run.watch`),
        - a table of 16 sample predictions logged as `wandb.Image` rows,
        - the trained model state dict, saved as a `wandb.Artifact`.

        You will need a free W&B account. Grab an API key from
        [wandb.ai/authorize](https://wandb.ai/authorize).
        """
    )
    return


@app.cell
def imports():
    import os

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    import wandb

    return (
        DataLoader,
        F,
        nn,
        os,
        torch,
        torchvision,
        transforms,
        wandb,
    )


@app.cell
def login_cell(mo, os, wandb):
    # `wandb.login()` is a no-op if you are already authenticated on this
    # machine (e.g. via `wandb login` on the command line or the WANDB_API_KEY
    # env var). Otherwise marimo will prompt for the key in the cell output.
    _logged_in = wandb.login() if not os.environ.get("WANDB_DISABLED") else False
    mo.md(
        f"**W&B login status:** {'authenticated' if _logged_in else 'not authenticated — `wandb.login()` will prompt at training time'}"
    )
    return


@app.cell
def hyperparams(mo):
    epochs = mo.ui.slider(start=1, stop=10, value=2, step=1, label="Epochs")
    batch_size = mo.ui.slider(
        start=32, stop=256, value=128, step=32, label="Batch size"
    )
    learning_rate = mo.ui.slider(
        start=1e-4,
        stop=1e-1,
        value=1e-3,
        step=1e-4,
        label="Learning rate",
        show_value=True,
    )
    project_name = mo.ui.text(
        value="mnist-marimo", label="W&B project", full_width=False
    )
    run_button = mo.ui.run_button(label="Start training run", kind="success")

    controls = mo.vstack(
        [
            mo.md("### Hyperparameters"),
            epochs,
            batch_size,
            learning_rate,
            project_name,
            run_button,
        ]
    )
    controls
    return batch_size, epochs, learning_rate, project_name, run_button


@app.cell
def device_cell(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mo.md(f"**Compute device:** `{device}`")
    return (device,)


@app.cell
def data(DataLoader, batch_size, torchvision, transforms):
    # MNIST mean and std, computed across the training set.
    _normalize = transforms.Normalize((0.1307,), (0.3081,))
    _transform = transforms.Compose([transforms.ToTensor(), _normalize])

    _data_root = "./.data"
    train_dataset = torchvision.datasets.MNIST(
        root=_data_root, train=True, download=True, transform=_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=_data_root, train=False, download=True, transform=_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size.value, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=0
    )
    return test_dataset, test_loader, train_dataset, train_loader


@app.cell
def model_def(F, nn, torch):
    class MNISTConvNet(nn.Module):
        """A small CNN: 2x (conv → ReLU → max-pool) → linear → logits."""

        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # After two 2x2 pools on 28x28 -> 7x7 feature maps.
            self.fc = nn.Linear(32 * 7 * 7, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, start_dim=1)
            return self.fc(x)

    def build_model() -> nn.Module:
        return MNISTConvNet()

    return MNISTConvNet, build_model


@app.cell
def training(
    F,
    batch_size,
    build_model,
    device,
    epochs,
    learning_rate,
    mo,
    project_name,
    run_button,
    test_dataset,
    test_loader,
    torch,
    train_loader,
    wandb,
):
    # Halt this cell until the user presses "Start training run". Without this
    # guard, marimo's reactive engine would kick off a training run every time
    # any upstream cell changes.
    mo.stop(
        not run_button.value,
        mo.md(
            "_Adjust the hyperparameters above, then press "
            "**Start training run** to begin._"
        ),
    )

    config = {
        "epochs": int(epochs.value),
        "batch_size": int(batch_size.value),
        "learning_rate": float(learning_rate.value),
        "architecture": "CNN",
        "dataset": "MNIST",
        "device": str(device),
    }

    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    with wandb.init(
        project=project_name.value,
        config=config,
        notes="Trained from marimo notebook (notebooks/marimo/mnist_wandb.py)",
    ) as run:
        # Stream parameter and gradient histograms to W&B every 50 steps.
        run.watch(model, criterion=criterion, log="all", log_freq=50)

        global_step = 0
        for epoch in mo.status.progress_bar(
            range(config["epochs"]),
            title="Training",
            subtitle=f"{config['epochs']} epoch(s) on {device}",
        ):
            # ---- Train ----
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                run.log(
                    {"train/loss": loss.item(), "epoch": epoch},
                    step=global_step,
                )
                global_step += 1

            # ---- Validate ----
            model.eval()
            correct = 0
            total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    val_loss_sum += F.cross_entropy(
                        logits, labels, reduction="sum"
                    ).item()
                    predicted = logits.argmax(dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            run.log(
                {
                    "val/accuracy": correct / total,
                    "val/loss": val_loss_sum / total,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # ---- Log 16 sample predictions ----
        sample_table = wandb.Table(
            columns=["image", "predicted", "actual", "correct"]
        )
        model.eval()
        with torch.no_grad():
            for i in range(16):
                image, label = test_dataset[i]
                logits = model(image.unsqueeze(0).to(device))
                predicted = int(logits.argmax(dim=1).item())
                sample_table.add_data(
                    wandb.Image(image, caption=f"true={label}, pred={predicted}"),
                    predicted,
                    int(label),
                    predicted == label,
                )
        run.log({"predictions": sample_table})

        # ---- Save the model as a versioned W&B artifact ----
        _model_path = "mnist_cnn.pt"
        torch.save(model.state_dict(), _model_path)
        artifact = wandb.Artifact(
            "mnist-cnn",
            type="model",
            metadata={
                **config,
                "final_val_accuracy": correct / total,
            },
        )
        artifact.add_file(_model_path)
        run.log_artifact(artifact)

        run_url = run.url
        final_accuracy = correct / total

    summary = mo.md(
        f"""
        ### Run complete

        - **Final validation accuracy:** `{final_accuracy:.4f}`
        - **W&B run:** [{run_url}]({run_url})

        Press **Start training run** again to launch another run with the
        current hyperparameters — each press creates a new W&B run.
        """
    )
    summary
    return final_accuracy, model, run_url


@app.cell(hide_code=True)
def outro(mo):
    mo.md(
        """
        ## What to look at in W&B

        Open the run URL printed above. In the W&B Models UI you can:

        - **Charts** — inspect the `train/loss`, `val/accuracy`, and
          `val/loss` curves.
        - **System** — parameter and gradient histograms produced by
          `run.watch(model, log="all")`.
        - **Tables** — the `predictions` table with 16 sample test images,
          predicted vs. true labels, and a `correct` boolean column you can
          filter on.
        - **Artifacts** — the `mnist-cnn` model artifact. Re-running the
          notebook produces a new version (`v1`, `v2`, …) automatically.

        To compare runs, change a hyperparameter (e.g. learning rate), press
        **Start training run** again, then in the W&B project select both
        runs and switch to the comparison view.
        """
    )
    return


if __name__ == "__main__":
    app.run()
