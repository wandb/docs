---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Organize models

Use model tags to organize registered models into categories and to search over those categories. You can organize models programmatically or interactively with the W&B App UI.




<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'W&B Python SDK', value: 'api'},
  ]}>
  <TabItem value="app">


1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select **View details** next to the name of the registered model you want to add a model tag to. 
    ![](/images/models/organize-models-model-reg-landing.png)
2. Scroll to the **Model card** section.
3. Click the plus button (**+**) next to the **Tags** field.
![](/images/models/organize-models-seleticon.png)
4. Type in the name for your tag or search for a pre-existing model tag.
    For example. the following image shows multiple model tags added to a registered model called **FineTuned-Review-Autocompletion**:

    ![](/images/models/model-tags-modelregview.png)


  </TabItem>
  <TabItem value="api">

Use the [`Api().artifact_collection()`](../../ref/python/public-api/api.md#artifact_collection) method to programmatically organize and update a model's description, name, and tags.

Specify `"model"` string for the `type_name` parameter and provide the name of the artifact collection for the `name` parameter. 

:::tip
The name of an Artifact collection consists of the entity that owns the artifact, the term `model-registry`, and the name of the artifact collection itself: `<entity>/model-registry/<artifact_collection_name>`.
:::

For example, the proceeding code snippet shows how to update the description, name, and tag of a model:

```python
import wandb 

name = "<entity>/model-registry/<artifact_collection_name>"
type_name = "model"
registered_model = wandb.Api().artifact_collection(type_name = type_name, \ 
                                                   name = name )
registered_model.description = "this is a new description"
registered_model.name = "Anomaly Detection Part 2"
registered_model.tags = ["test1", "test2"]
registered_model.save()
```


  </TabItem>
</Tabs>