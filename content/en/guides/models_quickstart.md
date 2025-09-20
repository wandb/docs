---
title: Get Started with W&B Models
weight: 2
---

Learn when and how to use W&B to track, share, and manage model artifacts in your machine learning workflows. This page covers logging experiments, generating reports, and accessing logged data using the appropriate W&B API for each task.

This tutorial uses the following:

* [W&B Python SDK]({{< relref "/ref/python" >}}) (`wandb.sdk`): to log and monitor experiments during training.
* [W&B Public API]({{< relref "/ref/python/public-api" >}}) (`wandb.apis.public`): to query and analyze logged experiment data.
* [W&B Reports and Workspaces API]({{< relref "/ref/wandb_workspaces" >}}) (`wandb.wandb-workspaces`): to create a report to summarize findings.

## Sign up and create an API key
To authenticate your machine with W&B, you must first generate an API key at [wandb.ai/authorize](https://wandb.ai/authorize). Copy the API key and store it securely.

## Install and import packages

Install the W&B library and some other packages you will need for this walkthrough.  

{{< code language="shell" source="/bluehawk_source/snippets/wandb_install.snippet.pip_install_wandb_packages.sh" >}}

Import W&B Python SDK:


{{< code language="python" source="/bluehawk_source/snippets/import_wandb.snippet.import_wandb.py" >}}


Next, import the Reports and Workspaces API:

{{< code language="python" source="/bluehawk_source/snippets/import_wandb.snippet.import_wandb_workspaces.py" >}}

Specify the entity of your team in the following code block:


```python
TEAM_ENTITY = "<Team_Entity>" # Replace with your team entity
PROJECT = "my-awesome-project"
```

## Train a model

The following code simulates a basic machine learning workflow: training a model, logging metrics, and saving the model as an artifact.

Use the W&B Python SDK (`wandb.sdk`) to interact with W&B during training. Log the loss using [`wandb.Run.log()`]({{< relref "/ref/python/experiments/run/#method-runlog" >}}), then save the trained model as an artifact using [`wandb.Artifact`]({{< relref "/ref/python/experiments/artifact.md" >}}) before finally adding the model file using [`Artifact.add_file`]({{< relref "/ref/python/experiments/artifact.md#add_file" >}}).

{{< code language="python" source="/bluehawk_source/snippets/models_quickart.snippet.train_model.py" >}}

The key takeaways from the previous code block are:
* Use `wandb.Run.log()` to log metrics during training.
* Use `wandb.Artifact` to save models (datasets, and so forth) as an artifact to your W&B project.

Now that you have trained a model and saved it as an artifact, you can publish it to a registry in W&B. Use [`wandb.Run.use_artifact()`]({{< relref "/ref/python/experiments/run/#method-runuse_artifact" >}}) to retrieve the artifact from your project and prepare it for publication in the Model registry. `wandb.Run.use_artifact()` serves two key purposes:
* Retrieves the artifact object from your project.
* Marks the artifact as an input to the run, ensuring reproducibility and traceability. See [Create and view lineage map]({{< relref "/guides/core/registry/lineage/" >}}) for details.

## View the training data in the dashboard

Log in to your account at https://wandb.ai/login

Under **Projects** you should see `my-awesome-project` (or whatever you used as a project name above). Click this to enter the workspace for your project. 

From here, you can see details about every run you've done. In this screenshot, the code was re-run several times, generating a number of runs, each of which is given a randomly-generated name. 

{{< img "/images/quickstart/quickstart_image.png" >}}


## Publish the model to the Model registry

To share the model with others in your organization, publish it to a [collection]({{< relref "/guides/core/registry/create_collection" >}}) using `wandb.Run.link_artifact()`. The following code links the artifact to the [core Model registry]({{< relref "/guides/core/registry/registry_types/#core-registry" >}}), making it accessible to your team.

{{< code language="python" source="/bluehawk_source/snippets/models_quickart.snippet.publish_model.py" >}}

After running `wandb.Run.link_artifact()`, the model artifact will be in the `DemoModels` collection in your registry. From there, you can view details such as the version history, [lineage map]({{< relref "/guides/core/registry/lineage/" >}}), and other [metadata]({{< relref "/guides/core/registry/registry_cards/" >}}). 

For additional information on how to link artifacts to a registry, see [Link artifacts to a registry]({{< relref "/guides/core/registry/link_version/" >}}).

## Retrieve model artifact from registry for inference

To use a model for inference, use `wandb.Run.use_artifact()` to retrieve the published artifact from the registry. This returns an artifact object that you can then use [`wandb.Artifact.download()`]({{< relref "/ref/python/experiments/artifact/#method-artifactdownload" >}}) to download the artifact to a local file.

{{< code language="python" source="/bluehawk_source/snippets/models_quickart.snippet.retrieve_model.py" >}}

For more information on how to retrieve artifacts from a registry, see [Download an artifact from a registry]({{< relref "/guides/core/registry/download_use_artifact/" >}}).

Depending on your machine learning framework, you may need to recreate the model architecture before loading the weights. This is left as an exercise for the reader, as it depends on the specific framework and model you are using. 

## Share your finds with a report

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

Create and share a [report]({{< relref "/guides/core/reports/_index.md" >}}) to summarize your work. To create a report programmatically, use the [W&B Report and Workspace API]({{< relref "/ref/wandb_workspaces/reports.md" >}}).

First, install the W&B Reports API:

```python
pip install wandb wandb-workspaces -qqq
```

The following code block creates a report with multiple blocks, including markdown, panel grids, and more. You can customize the report by adding more blocks or changing the content of existing blocks. 

The output of the code block prints a link to the URL report created. You can open this link in your browser to view the report. 

{{< code language="python" source="/bluehawk_source/snippets/models_quickart.snippet.share_report.py" >}}

For more information on how to create a report programmatically or how to create a report interactively with the W&B App, see [Create a report]({{< relref "/guides/core/reports/create-a-report.md" >}}) in the W&B Docs Developer guide. 

## Query the registry
Use the [W&B Public APIs]({{< relref "/ref/python/public-api/_index.md" >}}) to query, analyze, and manage historical data from W&B. This can be useful for tracking the lineage of artifacts, comparing different versions, and analyzing the performance of models over time.

The following code block demonstrates how to query the Model registry for all artifacts in a specific collection. It retrieves the collection and iterates through its versions, printing out the name and version of each artifact.

{{< code language="python" source="/bluehawk_source/snippets/models_quickart.snippet.query_registry.py" >}}

For more information on querying the registry, see the [Query registry items with MongoDB-style queries]({{< relref "/guides/core/registry/search_registry.md#query-registry-items-with-mongodb-style-queries" >}}).
