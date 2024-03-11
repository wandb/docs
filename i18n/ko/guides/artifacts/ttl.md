---
description: Time to live policies (TTL)
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Manage data retention with Artifact TTL policy

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb"/>

Schedule when artifacts are deleted from W&B with W&B Artifact time-to-live (TTL) policy. When you delete an artifact, W&B marks that artifact as a *soft-delete*. In other words, the artifact is marked for deletion but files are not immediately deleted from storage. For more information on how W&B deletes artifacts, see the [Delete artifacts](./delete-artifacts.md) page.

Check out [this](https://www.youtube.com/watch?v=hQ9J6BoVmnc) video tutorial to learn how to manage data retention with Artifacts TTL in the W&B App.

:::note
W&B deactivates the option to set a TTL policy for model artifacts linked to the Model Registry. This is to help ensure that linked models do not accidentally expire if used in production workflows.
:::
:::info
* Only team admins can view a [team's settings](../app/settings-page/team-settings.md) and access team level TTL settings such as (1) permitting who can set or edit a TTL policy or (2) setting a team default TTL.  
* If you do not see the option to set or edit a TTL policy in an artifact's details in the W&B App UI or if setting a TTL programmatically does not successfully change an artifact's TTL property, your team admin has not given you permissions to do so. 
:::


## Define who can edit and set TTL policies
Define who can set and edit TTL policies within a team. You can either grant TTL permissions only to team admins, or you can grant both team admins and team members TTL permissions. 

:::info
Only team admins can define who can set or edit a TTL policy.
:::

1. Navigate to your team’s profile page.
2. Select the **Settings** tab.
3. Navigate to the **Artifacts time-to-live (TTL) section**.
4. From the **TTL permissions dropdown**, select who can set and edit TTL policies.  
5. Click on **Review and save settings**. 
6. Confirm the changes and select **Save settings**. 

![](/images/artifacts/define_who_sets_ttl.gif)

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

The preceding code snippet sets the TTL policy for the artifact to 30 days. In other words, W&B deletes the artifact after 30 days.

### Set or edit a TTL policy after you create an artifact
Use the W&B App UI or the W&B Python SDK to define a TTL policy for an artifact that already exists.

:::note
When you modify an artifact's TTL, the time the artifact takes to expire is still calculated using the artifact's `createdAt` timestamp.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

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

The preceding code example sets the TTL policy to two years.

  </TabItem>
  <TabItem value="app">

1. Navigate to your W&B project in the W&B App UI.
2. Select the artifact icon on the left panel.
3. From the list of artifacts, expand the artifact type you 
4. Select on the artifact version you want to edit the TTL policy for.
5. Click on the **Version** tab.
6. From the dropdown, select **Edit TTL policy**.
7. Within the modal that appears, select **Custom** from the TTL policy dropdown.
8. Within the **TTL duration** field, set the TTL policy in units of days.
9. Select the **Update TTL** button to save your changes.

![](/images/artifacts/edit_ttl_ui.gif)

  </TabItem>
</Tabs>


### Set default TTL policies for a team

:::info
Only team admins can set a default TTL policy for a team.
:::

Set a default TTL policy for your team. Default TTL policies apply to all existing and future artifacts based on their respective creation dates. Artifacts with existing version-level TTL policies are not affected by the team's default TTL.

1. Navigate to your team’s profile page.
2. Select the **Settings** tab.
3. Navigate to the **Artifacts time-to-live (TTL) section**.
4. Click on the **Set team's default TTL policy**.
5. Within the **Duration** field, set the TTL policy in units of days.
6. Click on **Review and save settings**.
7/ Confirm the changes and then select **Save settings**. 

![](/images/artifacts/set_default_ttl.gif)



## Deactivate a TTL policy
Use the W&B Python SDK or W&B App UI to deactivate a TTL policy for a specific artifact version.
<!-- 
:::note
Artifacts with a disabled TTL will not inherit an artifact collection's TTL. Refer to (## Inherit TTL Policy) on how to delete artifact TTL and inherit from the collection level TTL.
::: -->

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="python">

1. [Fetch your artifact](./download-and-use-an-artifact.md).
2. Set the artifact's `ttl` attribute to `None`.
3. Update the artifact with the [`save`](../../ref/python/run.md#save) method.


The following code snippet shows how to turn off a TTL policy for an artifact:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```


  </TabItem>
  <TabItem value="app">

1. Navigate to your W&B project in the W&B App UI.
2. Select the artifact icon on the left panel.
3. From the list of artifacts, expand the artifact type you 
4. Select on the artifact version you want to edit the TTL policy for.
5. Click on the Version tab.
6. Click on the meatball UI icon next to the **Link to registry** button. 
7. From the dropdown, select **Edit TTL policy**.
8. Within the modal that appears, select **Deactivate** from the TTL policy dropdown.
9. Select the **Update TTL** button to save your changes.

![](/images/artifacts/remove_ttl_polilcy.gif)

  </TabItem>
</Tabs>



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
