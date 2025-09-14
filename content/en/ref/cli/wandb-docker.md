---
title: wandb docker
---

Run your code in a docker container.

W&B docker lets you run your code in a docker image ensuring wandb is configured. It adds the WANDB_DOCKER and WANDB_API_KEY environment variables to your container and mounts the current directory in /app by default. You can pass additional args which will be added to `docker run` before the image name is declared, we'll choose a default image for you if one isn't passed:

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

By default, we override the entrypoint to check for the existence of wandb and install it if not present. If you pass the --jupyter flag we will ensure jupyter is installed and start jupyter lab on port 8888. If we detect nvidia-docker on your system we will use the nvidia runtime. If you just want wandb to set environment variable to an existing docker run command, see the wandb docker-run command.

## Usage

```bash
wandb docker [DOCKER_RUN_ARGS] [DOCKER_IMAGE] [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `DOCKER_RUN_ARGS` | No description available | No |
| `DOCKER_IMAGE` | No description available | No |

## Options

| Option | Description |
| :--- | :--- |
| `--nvidia` | Use the nvidia runtime, defaults to nvidia if nvidia-docker is present (default: False) |
| `--digest` | Output the image digest and exit (default: False) |
| `--jupyter` | Run jupyter lab in the container (default: False) |
| `--dir` | Which directory to mount the code in the container (default: /app) |
| `--no-dir` | Don't mount the current directory (default: False) |
| `--shell` | The shell to start the container with (default: /bin/bash) |
| `--port` | The host port to bind jupyter on (default: 8888) |
| `--cmd` | The command to run in the container |
| `--no-tty` | Run the command without a tty (default: False) |
