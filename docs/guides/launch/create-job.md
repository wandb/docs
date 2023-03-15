# Create a job

A W&B Job is a definition of a computational process. You can think of jobs as a ‘run template’. A job is populated with contextual information about the run it is created from such as the source code, software dependencies, hyperparameter configuration, and more. For more information, see the View Job details section [LINK].

There are three ways to create a job:

1. Log a code artifact
2. Associate a Git repository
3. Set the `WANDB_DOCKER` environment variable

W&B automatically checks your run to see if it can create a job. W&B determines how to create a job based on the following logic:

1. If a remote Git repository is present, create a GitHub sourced job.
2. If `run.log_code()` is called, create an artifact sourced job.
3. Lastly, if the `WANDB_DOCKER` environment variable is set, create an image sourced job.

:::info
Set `wandb.Settings(disable_git=True)` within your script to prevent W&B from creating a git sourced job.

Remove `run.log_code()` from your script to prevent W&B from creating an artifact sourced job.
:::

Based on your use case, read the sections below to create a job using code artifacts, associating a GitHub repo, or setting an environment variable.

## Log a code artifact
Create a W&B Run similarly to how you normally would when you create a W&B Experiment or Run. Add `wandb.init()` and `wandb.log()` to start a new W&B Run to track and log to W&B. Ensure you specify `run.log_code()`. W&B will create a Job when you specify `run.log_code()`. 

### Example Python script
The following code shows an example of how W&B can create a job for you from a Python script that logs code:

```python
# canonical_job_example.py

import random
import wandb

def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="job_example",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        })

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})
    
    run.log_code()


run_training_run(epochs=10, lr=0.01)
```

Within your terminal type in the following command:

```bash
python canonical_job_example.py
```

 ## Create a job from a GitHub repository
W&B Launch can automatically create a W&B Job if you have a GitHub repository with a Python script that creates a W&B Run. 

:::info
Unlike the Log a code artifact section [LINK], you do not need to specify `run.log_code()` within your Python script. 
:::

1. Navigate to your personal or team settings page at [https://wandb.ai/settings](https://wandb.ai/settings) .


2. Create a GitHub repo with your Python script if you have not done so already. 
3. Run your Python script.

### Example GitHub repo and Python script
For example, suppose we create a GitHub repository called “demo_launch” with a README.md file and a Python script called `canonical_job_example.py`:


![](/images/launch/example_github_repo.png)

The `canonical_job_example.py` script contains this code:

```python
import random
import wandb


def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="job_example",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        })

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})


run_training_run(epochs=10, lr=0.01)
```

Next, we run the Python script to create a W&B Run that is tracked by W&B. In this example, we run our script locally on our machine:
```bash
python canonical_job_example.py
```
That's it! You can see the job created for you on the W&B App.

1. Navigate to your project on the W&B App. 
2. Select the Jobs icon on the left panel. 

![](/images/launch/job_page_github_repo_example_zoom.png)

The name of the job will contain the prefix `job-` with the GitHub URL and path to script as the suffix: `job-<git-remote-url>-<path-to-script>`. 

For more information on how job names are created, see Job naming conventions[LINK].

 ## Create a job with your own Docker container.

W&B Launch can automatically create a job if you use a Docker container that satisfies the following requirements:

1. The directory with your `Dockerfile` contains a Python script that creates a W&B Run.
2. You specify the `WANDB_DOCKER` environment variable and you pass your W&B Api Key when you create the Docker container.

### Example Dockerfile and training script
The following code shows an example of how to create a Docker container that satisfies the requirements for W&B to automatically create a job.

1. Copy the following code and save it as a `Dockerfile`

```
FROM python:3.8
COPY . /src
WORKDIR /src
RUN pip install wandb
ENTRYPOINT [ "env", "python3", "train.py"]
```

2. Copy the following code and save it into a Python script called `train.py`: 
```python
# train.py

import random
import wandb

def run_training_run(epochs, lr):
    print(f"Training for {epochs} epochs with learning rate {lr}")

    run = wandb.init(
        # Set the project where this run will be logged
        project="job_example",
        # Track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        })

    offset = random.random() / 5
    print(f"lr: {lr}")

    for epoch in range(2, epochs):
        # simulating a training run
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, acc={acc}, loss={loss}")
        wandb.log({"acc": acc, "loss": loss})
		


run_training_run(epochs=10, lr=0.01)
```

3. Build the docker image:
```bash
docker build . -t launch-container-job
```

4. Create the docker container defined in the previous step with `docker run`. Set the `WANDB_DOCKER` environment variable to a docker image digest to enable restoring of runs and your W&B API key with the following:

```bash
docker run WANDB_API_KEY=... WANDB_DOCKER=<image-name> <image-name>
```

For more information on optional W&B Environment variables, see the [Environment Variables](../track/environment-variables.md) page.

## Job naming conventions

By default, W&B automatically creates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). The following table describes the job naming convention used for each job source:

| Source | Naming convention |
| ------ | ------ |
| GitHub | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>` |
| Docker image | `job-<image-name>` |