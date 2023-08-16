---
description: TTL
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Manage data retention with TTL
Use a time-to-live (TTL) policy to automatically remove an artifact version from W&B.

:::note
W&B does not support setting TTL policies for artifacts that are linked to the W&B Model Registry.
:::


## Create a TTL policy
Set a TTL policy for an artifact either when you create the artifact or retroactively after the artifact is created.


For all the code snippets below, replace the content wrapped in `<>` with your information to use the code snippet. 

### Set a TTL policy when you create an artifact
Use the W&B Python SDK to define a TTL policy when you create an artifact version. TTL policies are defined in days.    

:::tip
Defining a TTL policy when you create an artifact version is similar to how you normally [create an artifact](./construct-an-artifact.md). With the exception that you pass in a time delta to the artifact's `ttl_duration` attribute.
:::

The steps are as follows: 

1. [Create an artifact](./construct-an-artifact.md).
2. [Add content to the artifact](./construct-an-artifact.md#add-files-to-an-artifact) such as files, a directory, or a reference.
3. Define a TTL time limit with the [`datetime`](https://docs.python.org/3/library/datetime.html) data type part of Python's standard library. Pass this time delta to the artifact's `ttl_duration` attribute. 
4. [Log the artifact](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server).

The following code snippet demonstrates how to create an artifact and set a TTL policy. 

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity='<your-entity>')
art = wandb.Artifact(name="<artifact-name>", type="<type>")
art.add_file("my_file")

art.ttl = timedelta(days=30) # Set TTL policy
run.log_artifact(art)
```


### Set or edit a TTL policy after you create an artifact
Use the W&B Python SDK to define a TTL policy for an artifact that already exists.

1. [Download your artifact](./download-and-use-an-artifact.md).
2. Pass in a time delta to the artifact's `ttl_duration` attribute. 
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.


The following code snippet shows how to retrieve, view and set a new TTL policy:
```python
import wandb
from datetime import timedelta

art = run.use_artifact(artifact_or_name="<entity/project/your-artifact-name:alias>", type="<type>")
art.ttl_duration = timedelta(days=365*7) # pass in a new TTL
art.save()
```


## Deactivate a TTL policy
Use the W&B Python SDK to deactivate a TTL policy.

1. [Download your artifact](./download-and-use-an-artifact.md).
2. Set the artifact's `ttl_duration` attribute to `None`.
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.


The following code snippet shows how to deactivate a TTL policy for an artifact:
```python
art = run.use_artifact(artifact_or_name="<entity/project/your-artifact-name:alias>", type="<type>")
art.ttl_duration = None
art.save()
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

Use a print statement to view a TTL policy for an artifact. The following example shows how to retrieve an artifact and view its TTL policy:

```python
art = run.use_artifact(artifact_or_name="<entity/project/your-artifact-name:alias>", type="<type>")
print(art.ttl_duration)
```

  </TabItem>
  <TabItem value="app">


View a TTL policy for an artifact with the W&B App UI.

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home).
2. Go to your W&B Project.
3. Within your project, select the Artifacts tab in the left sidebar.
4. Click on a collection.

Within the collection view you can see all of the versions for the selected artifact. Within the `Time to Live` column you will see the TTL policy assigned to that artifact version. 

[INSERT IMAGE]

  </TabItem>
</Tabs>

