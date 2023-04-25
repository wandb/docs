---
slug: /guides/models
description: Manage the model lifecycle from training to production
---

# Model Registry

Use Model Registry to organize your best model versions for a given task. Easily track your models in their various stages of maturity: from staging to production, and much more.  Use the Model Registry App UI to see a history of all changes, including who moved a model to production

![](/images/models/model_registry_landing_page.png)

The image above shows the Model Registry W&B App UI. The left panel lists models registered to this user's account. On the right panel there is a **Model Overview** that describes [insert]. On the bottom right we can see different model versions of the model in the **Versions** section.



<!-- ### Watch the 1 minute video walkthrough -->

<!-- {% embed url="https://www.youtube.com/watch?v=jy9Pk9riwZI" %} -->

## How it works

The following Python code demonstrates a typical training script that uses W&B. However, this demo script also uses API calls that will track, register, and store a model into the model registry.


* Line 4: Create a W&B run object like you normally would.
* Line 16 - 18: Create a _model artifact_. (Specify 'model' as the `type` when you create the artifact instance). 
* Line 20: Create a _model version_. Log the model version with `log_artifact()`.
* Line 24-28: Link the model artifact to the model registry with `link_artifact()`.


```python showLineNumbers
import wandb
import random

run = wandb.init(project="Models_Quickstart")

# canonical training loop
for i in range(5):
    run.log({"acc": 0.91, "loss": 0.12})

    # Serialize the model
    f = open('my-model.h5', 'w')
    f.write(str(random.random())) # Imaginary model
    f.close()

    # Save the model checkpoint to W&B
    best_model = wandb.Artifact(
        f"model_{run.id}", type='model'
        )
    best_model.add_file('my-model.h5')
    run.log_artifact(best_model, aliases=["best"])


# Link the model to the Model Registry
run.link_artifact(
    artifact=best_model, 
    target_path='model-registry/Quickstart Registered Model', 
    aliases=['staging']
    )

run.finish()
```

Select the link URL printed from the W&B run output or navigate to your W&B project to view a dashboard similar to the image posted above. For more information about Model Registry terms, see the [Concepts](./model-management-concepts.md) page.


## How to get started
* Read [Model Management Concepts](./model-management-concepts.md) for more information on basic Model Management concepts.
* Follow the steps in the [Walkthrough](./walkthrough.md) to learn how to use Model Registry.


<!-- ## Model Registry Features

### Model Versioning

Iterate to get the best model version for a task, and catalog all the changes along the way.

* Track every model version in a central repository
* Browse and compare model versions
* Capture training metrics and hyperparameters

### Model Lineage

Document and reproduce the complete pipeline of model training and evaluation.

* Identify the exact dataset version the model trained on
* Restore the training code, including git commit and diff patch
* Get back to the model’s hyperparameters and other metadata for reproducibility
* Dig in to upstream jobs that can affect model performance

### Model Lifecycle

Manage the process as a model moves from training through staging to production.

* Highlight the best model versions that are being evaluated for production
* Communicate where a model version is in the process — staging, production etc
* Review the history of model versions that moved through each stage

## Model Registry Pilot Limits

This new feature is now turned on for all users to try for free, up to:

* 5 Registered Models, with unlimited versions linked to each model
* 10 most recent steps of Action History shown in the UI for each registered model -->




