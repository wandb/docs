---
description: TTL
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Manage data retention with Artifact TTL policy

Schedule when artifacts are deleted from W&B with W&B Artifact time-to-live (TTL) policy.

:::note
W&B deactivates the option to set a TTL policy for model artifacts linked to the Model Registry. This is to help ensure that models are not deleted by mistake.
:::


## Create a TTL policy
Set a TTL policy for an artifact either when you create the artifact or retroactively after the artifact is created.


For all the code snippets below, replace the content wrapped in `<>` with your information to use the code snippet. 

### Set a TTL policy when you create an artifact
Use the W&B Python SDK to define a TTL policy when you create an artifact. TTL policies are typically defined in days.    

:::tip
Defining a TTL policy when you create an artifact is similar to how you normally [create an artifact](./construct-an-artifact.md). With the exception that you pass in a time delta to the artifact's `ttl` attribute.
:::

The steps are as follows: 

1. [Create an artifact](./construct-an-artifact.md).
2. [Add content to the artifact](./construct-an-artifact.md#add-files-to-an-artifact) such as files, a directory, or a reference.
3. Define a TTL time limit with the [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) data type that is part of Python's standard library.
4. [Log the artifact](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server).

The following code snippet demonstrates how to create an artifact and set a TTL policy. 

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # Set TTL policy
run.log_artifact(artifact)
```

The code snippet above sets the TTL policy for the artifact to 30 days. In other words, the artifact is deleted by W&B shortly after 30 days.

### Set or edit a TTL policy after you create an artifact
Use the W&B Python SDK to define a TTL policy for an artifact that already exists.

1. [Fetch your artifact](./download-and-use-an-artifact.md).
2. Pass in a time delta to the artifact's `ttl` attribute. 
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.


The following code snippet shows how to set a TTL policy for an artifact:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # Delete in two years
artifact.save()
```

In the above example, the TTL policy is set to two years.

:::note
When an artifact's TTL is modified, the time the artifact takes to expire is still calculated using the artifact's `createdAt` timestamp.
:::

<!-- ## Inherit TTL policy [For incoming feature Artifact Collection level TTL]
Let an artifact, that does not have a TTL policy, inherit the TTL policy of the artifact collection it belongs to. An artifact can only inherit a TTL policy from the artifact collection it belongs to if the artifact collection possesses a TTL policy.  Otherwise, no TTL policy is inherited.

:::note
An artifact will not inherit a TTL policy from its artifact collection if a TTL policy already exists for that artifact.
:::

1. [Fetch your artifact](./download-and-use-an-artifact.md).
2. Set the artifact's `ttl` attribute to the constant `wandb.ArtifactTTL.INHERIT`.
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = wandb.ArtifactTTL.INHERIT
artifact.save()
``` -->

## Deactivate a TTL policy
Use the W&B Python SDK to deactivate a TTL policy.
<!-- 
:::note
Artifacts with a disabled TTL will not inherit an artifact collection's TTL. Refer to (## Inherit TTL Policy) on how to delete artifact TTL and inherit from the collection level TTL.
::: -->

1. [Fetch your artifact](./download-and-use-an-artifact.md).
2. Set the artifact's `ttl` attribute to `None`.
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.


The following code snippet shows how to disable a TTL policy for an artifact:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```

## View TTL policies
View TTL policies for artifacts with the Python SDK or with the W&B App UI.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

Use a print statement to view an artifact's TTL policy. The following example shows how to retrieve an artifact and view its TTL policy:

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```

  </TabItem>
  <TabItem value="app">


View a TTL policy for an artifact with the W&B App UI.

1. Navigate to the W&B App at [https://wandb.ai](https://wandb.ai).
2. Go to your W&B Project.
3. Within your project, select the Artifacts tab in the left sidebar.
4. Click on a collection.

Within the collection view you can see all of the artifacts in the selected collection. Within the `Time to Live` column you will see the TTL policy assigned to that artifact. 

![](/images/artifacts/ttl_collection_panel_ui.png)

  </TabItem>
</Tabs>
