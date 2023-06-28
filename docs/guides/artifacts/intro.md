---
slug: /guides/artifacts
description: >-
  Overview of what W&B Artifacts are, how they work, and how to get started
  using W&B Artifacts.
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';

# Artifacts

Try in product | [Try in colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb)

Use Artifacts to track and version any serialized data as the inputs and outputs of your W&B Runs. For example, a model training Run might take in a dataset as input and trained model as output. In addition to logging hyper-parameters and metadata to a W&B Run, you can use an Artifact to log the dataset used to train the model as input and the resulting model checkpoints as outputs. You will always be able answer the question “what version of my dataset was this model trained on”.

In summary, with W&B Artifacts, you can:
* [View where a model came from, including data it was trained on](./explore-and-traverse-an-artifact-graph.md).
* [Version every dataset change or model checkpoint](./create-a-new-artifact-version.md).
* [Easily reuse models and datasets across your team](./download-and-use-an-artifact.md).

![](/images/artifacts/artifacts_landing_page2.png)

The image above...[insert]

## Show me the code

Create an artifact with three lines of code:
1. Create an artifact with the `wandb.Artifact` API.
2. Add one or more files, such as a model file or dataset, to your artifact. 
3. Log your artifact to W&B.


```python showLineNumbers
artifact = wandb.Artifact('my_data', type='dataset')
artifact.add_reference(path_to_data) # Tracks data in a bucket
wandb.log_artifact(artifact) # Logs the artifact version "my_data:v0"
```

## [Temp Call to action]