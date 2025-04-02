---
title: API Walkthrough
weight: 1
---

Learn how to track, share, and manage model artifacts in your machine learning workflows with W&B. This notebook covers using W&B to log experiments, generate reports, and access logged data via the W&B Public API.

You'll use the following W&B packages:

* W&B Python SDK (`wandb.sdk`): Log and monitor experiments during training.
* W&B Public API (`wandb.apis.public`): Query and analyze logged experiment data.
* W&B Reports API (`wandb.wandb-workspaces`): Create reports to summarize findings.

## Sign up and create an API key
To authenticate your machine with W&B, generate an API key from your user profile or at wandb.ai/authorize. Copy the API key and store it securely.

## Install and import packages

Install the W&B library and some other packages you will need for this walkthrough.  



```python
!pip install wandb
```

Import W&B package:


```python
import wandb
```

Specify the entity of your team in the following code block:


```python
TEAM_ENTITY = "<Team_Entity>" # Replace with your team entity
PROJECT = "my-awesome-project"
```

## Train model

The following code simulates a basic machine learning workflow: training a model, logging metrics, and saving the model as an artifact.

You'll use the W&B Python SDK (`wandb.sdk`) to interact with W&B during training. Log the loss using [`wandb.log`](https://docs.wandb.ai/ref/python/run/#log) method, then save the trained model as an artifact. Create an artifact with [`wandb.Artifact`](https://docs.wandb.ai/ref/python/artifact/) and add the model file using [`Artifact.add_file`](https://docs.wandb.ai/ref/python/artifact/#add_file).


```python
import random # For simulating data

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

# Initialize W&B run
run = wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config)

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

# Unique name for the model artifact
model_artifact_name = f"model-{wandb.run.id}"  

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

# Explicitly tell W&B to finish the run
run.finish()
```

The key takeaways from the previous code block are:
* Use `wandb.log` to log metrics during training.
* Use `wandb.Artifact` to save models (datasets, and so forth) as an artifact using W&B.

> Note: This example uses `wandb.init()` without a context manager for simplicity. It is considered best practice to use a context manager (`with wandb.init() as run:`) to ensure that the run is properly closed after logging and avoids leaving open runs that can consume resources.

Since the model is saved to W&B (in the project specified in `wandb.init`), use [`wandb.use_artifact`](https://docs.wandb.ai/ref/python/run/#use_artifact) to retrieve the artifact and prepare it for publication in the Model registry.

`wandb.use_artifact` serves two key purposes:
* Retrieves the artifact object.
* Marks the artifact as an input to the run, ensuring reproducibility and traceability. See [Create model lineage map](https://docs.wandb.ai/guides/core/registry/model_registry/model-lineage/) for details.

## Publish the model to the Model registry
To share the model with others in your organization, publish it to a collection using `wandb.link_artifact`. This links the artifact to the [core Model registry](https://docs.wandb.ai/guides/core/registry/registry_types/#core-registry), making it accessible to your team.


```python
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
```

Within the Registry App, you should see the model artifact you just published within the DemoModels collection in the Model registry. You can click on it to view its details, including the version history, lineage map, and other metadata.

## Retrieve model artifact from registry for inference

To use a model for inference in a different environment or project, download the artifact from the Model registry using the W&B Public API.


```python
REGISTRY_NAME = "Model"  # Name of the registry in W&B
COLLECTION_NAME = "DemoModels"  # Name of the collection in the registry
VERSION = 0

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")
```

Use `wandb.use_artifact` to retrieve the published artifact from the Model registry. This returns an artifact object that you can then use the [`Artifact.download`](https://docs.wandb.ai/ref/python/artifact/#download) method to download the artifact.


```python
run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()
```

Depending on your machine learning framework, you may need to recreate the model architecture before loading the weights. This is left as an exercise for the reader, as it depends on the specific framework and model you are using. 

## Share your finds with a report

Create and share a report to summarize your work. To create a report programmatically, use the W&B Reports API.

First, install the W&B Reports API:


```python
!pip install wandb wandb-workspaces -qqq
```

Next, import the package:


```python
import wandb_workspaces.reports.v2 as wr
```

The following code block creates a report with multiple blocks, including markdown, panel grids, and more. You can customize the report by adding more blocks or changing the content of existing blocks. 

The output of the code block prints a link to the URL report created. You can open this link in your browser to view the report. 


```python
experiment_summary = """This is a summary of the experiment conducted to train a simple model using W&B."""
dataset_info = """The dataset used for training consists of synthetic data generated by a simple model."""
model_info = """The model is a simple linear regression model that predicts output based on input data with some noise."""

report = wr.Report(
    project=PROJECT,
    entity=TEAM_ENTITY,
    title="My Awesome Model Training Report",
    description=experiment_summary,
    blocks= [
        wr.TableOfContents(),
        wr.H2("Experiment Summary"),
        wr.MarkdownBlock(text=experiment_summary),
        wr.H2("Dataset Information"),
        wr.MarkdownBlock(text=dataset_info),
        wr.H2("Model Information"),
        wr.MarkdownBlock(text = model_info),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(title="Train Loss", x="Step", y=["loss"], title_x="Step", title_y="Loss")
                ],
            ),  
    ]

)

# Save the report to W&B
report.save()
```

For more information on how to create a report programmatically or how to create a report interactively with the W&B App, see [Create a report](https://docs.wandb.ai/guides/reports/create-a-report/) in the W&B Docs Developer guide. 

## Query the registry
Suppose time has elapsed since someone on your team created an artifact version and published it to a collection in a registry. You are unsure which artifacts are available in the registry. You can use the [W&B Public APIs](https://docs.wandb.ai/ref/python/public-api/) to query, analyze, and manage historical data from W&B. 

The following code block demonstrates how to query the Model registry for all artifacts in a specific collection. It retrieves the collection and iterates through its versions, printing out the name and version of each artifact.


```python
import wandb

# Initialize wandb API
api = wandb.Api()

# Find all artifact versions that contains the string `model` and 
# has either the tag `text-classification` or an `latest` alias
registry_filters = {
    "name": {"$regex": "model"}
}

# Use logical $or operator to filter artifact versions
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# Returns an iterable of all artifact versions that match the filters
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# Print out the name, collection, aliases, tags, and created_at date of each artifact found
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

For more information on querying the registry, see the [Query registry items with MongoDB-style queries](https://docs.wandb.ai/guides/core/registry/search_registry/#query-registry-items-with-mongodb-style-queries).

## Next steps


