---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Organize collections or artifact versions

Use tags to organize your collections or artifact versions within your registry. You can add a tag to a collection or artifact version programmatically with the W&B Python SDK or interactively with the W&B App UI.

## Add a tag to a collection

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of a collection
4. Within the collection card, click on the plus icon (**+**) next to the **Tags** field and type in the name of the tag
5. Hit **Enter** on your keyboard

![](/images/registry/add_tag_collection.gif)

## View tags that belong to a collection

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of a collection
4. Within the collection card you will one or more blue rectangles next to the **Tags** field

![](/images/registry/tag_collection_selected.png)

You can also view tags added to a collection when you select a registry card. Within the registry card, you will see one or more collections. If a tag was added a collection, you will see that tag appear as a blue rectangle next to the name of the collection.

![](/images/registry/tag_collection.png)


## Remove a tag from a collection

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of a collection
4. Within the collection card, hover your mouse over the name of the tag you want to remove
5. Click on the **X** icon

## Add a tag to an artifact version

<Tabs
  defaultValue="app_ui"
  values={[
    {label: 'W&B App UI', value: 'app_ui'},
    {label: 'Python SDK', value: 'python'},
  ]}>
  <TabItem value="app_ui">

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of the collection you want to add a tag to
4. Scroll down to **Versions**
5. Click **View** next to an artifact version
6. Within the **Version** tab, click on the plus icon (**+**) next to the **Tags** field and type in the name of the tag
7. Hit **Enter** on your keyboard

![](/images/registry/add_tag_linked_artifact_version.gif)


  </TabItem>  
  <TabItem value="python">

Fetch the artifact version you want to add or update a tag to. Once you have the artifact version, you can access the artifact object's `tag` attribute to add or modify tags to that artifact. Pass in one or more tags as list to the artifacts `tag` attribute.

Like other artifacts, you can fetch an artifact from W&B without creating a run or you can create a run and fetch the artifact within that run. In either case, ensure to call the artifact object's `save` method to update the artifact on the W&B servers.

Based on your use case, copy and paste one of the code cells below to add or modify an artifact version's tag. Ensure to replace the values in `<>` with your own.


The proceeding code snippet shows how to fetch an artifact and add a tag without creating a new run:
```python title="Add a tag to an artifact version without creating a new run"
import wandb

api = wandb.Api()

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = api.artifact(name = artifact_name, type = "<type>")
artifact.tags = ["tag2"] # Provide one or more tags in a list
artifact.save()
```


The proceeding code snippet shows how to fetch an artifact and add a tag by creating a new run:
```python title="Add a tag to an artifact version during a run"
import wandb

run = wandb.init(entity = "<entity>", project="<project>", job_type="<job-type>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
artifact.tags = ["tag2"] # # Provide one or more tags in a list
artifact.save()
```


  </TabItem>
</Tabs>

## View tags that belong to an artifact version


<Tabs
  defaultValue="app_ui"
  values={[
    {label: 'W&B App UI', value: 'app_ui'},
    {label: 'Python SDK', value: 'python'},
  ]}>
  <TabItem value="app_ui">

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of the collection you want to add a tag to
4. Scroll down to **Versions** section
5. Within the **Tags** column, you will see tags that were added to a each artifact version

![](/images/registry/tag_artifact_version.png)


  </TabItem>
  <TabItem value="python">

Fetch the artifact version to view its tags. Once you have the artifact version, you can view tags that belong to that artifact by viewing the artifact object's `tag` attribute.

Like other artifacts, you can fetch an artifact from W&B without creating a run or you can create a run and fetch the artifact within that run.

Based on your use case, copy and paste one of the code cells below to view an artifact version's tags. Ensure to replace the values in `<>` with your own.

The proceeding code snippet shows how to fetch and view an artifact version's tags without creating a new run:
```python title="Add a tag to an artifact version without creating a new run"
import wandb

api = wandb.Api()

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = api.artifact(name = artifact_name, type = "<type>")
print(artifact.tags)
```


The proceeding code snippet shows how to fetch and view artifact version's tags by creating a new run:
```python title="Add a tag to an artifact version during a run"
import wandb

run = wandb.init(entity = "<entity>", project="<project>", job_type="<job-type>")

artifact_name = f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"

artifact = run.use_artifact(artifact_or_name = artifact_name)
print(artifact.tags)
```



  </TabItem>
</Tabs>


## Remove a tag from an artifact version

1. Navigate to the W&B Registry at https://wandb.ai/registry
2. Click on a registry card
3. Click **View details** next to the name of the collection you want to add a tag to
4. Scroll down to **Versions**
5. Click **View** next to an artifact version
6. Within the **Version** tab, hover your mouse over the name of the tag
7. Click on the **X** icon 