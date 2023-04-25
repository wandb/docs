---
slug: /guides/models
description: Manage the model lifecycle from training to production
---

# Model Registry

Use Model Registry to organize your best model versions for a given task. Easily track your models in their various stages of maturity: from staging to production, and much more.  Use the Model Registry App UI to see a history of all changes, including who moved a model to production

![](/images/models/model_registry_landing_page.png)


For more information about the Model Registry UI, see [insert future docs].

<!-- ### Watch the 1 minute video walkthrough -->

<!-- {% embed url="https://www.youtube.com/watch?v=jy9Pk9riwZI" %} -->

## How it works
Use the W&B Python SDK to train, log, and use model versions. You can use the W&B App or the Python SDK to register or link models.


For example, the following Python code demonstrates a typical training script that uses W&B. However, this demo script also calls W&B Python SDK APIs that will create a model artifact, make a model version, and register the model to the Model Registry.

Copy the sample Python script below and run it in a Python script or in a Jupyter Notebook:


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
    # highlight-start
    best_model = wandb.Artifact(
        f"model_{run.id}", type='model'
        )
    best_model.add_file('my-model.h5')
    run.log_artifact(best_model, aliases=["best"])
    # highlight-end

# Link the model to the Model Registry
# highlight-start
run.link_artifact(
    artifact=best_model, 
    target_path='model-registry/Quickstart Model Registry', 
    aliases=['staging']
    )
# highlight-end

run.finish()
```
Where:

* Line 4: Create a W&B run object like you normally would.
* Line 16 - 18: Create a _model artifact_. (Specify 'model' as the `type` when you create the artifact instance). 
* Line 20: Create a _model version_. Log the model version with `log_artifact()`.
* Line 24-28: Link the model artifact to the model registry with `link_artifact()`.

Select the link URL printed from the W&B run output or navigate to your W&B project to view a dashboard similar to the image posted above. For more information about Model Registry terms, see the [Concepts](./model-management-concepts.md) page.


## How to get started
* Follow the steps in the [Walkthrough](./walkthrough.md) to learn how to use Model Registry.
* Read [Model Management Concepts](./model-management-concepts.md) for more information on basic Model Management concepts.
* [INSERT]
* [INSERT]






