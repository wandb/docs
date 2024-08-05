---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link an artifact version to a registry

Programmatically or interactively link artifact versions to a registry. 

W&B recommends that you check which artifact types the registry you want to link artifact versions to permits. Each registry controls the types of artifacts that can be linked to that registry.

:::info
The term "type" refers to the artifact object type. When you create an artifact object ([`wandb.Artifact`](../../ref/python/artifact.md)), or log an artifact ([`wandb.run.log_artifact`](../../ref/python/run.md#log_artifact)), you specify a type for the `type` parameter. 
<!-- If you are familiar with Python, you can think of artifact types in W&B as having similar functions as Python data types.  -->
:::

As an example, by default, the Model registry only permits artifacts objects that have a "model" type. W&B will not permit you to link a dataset artifact type object if you try to link it to the Model registry.

:::info
When you link an artifact to a registry, this "publishes" that artifact to that registry. Any user that has access to that registry can access linked artifact versions when you link an artifact to a collection.

In other words, linking an artifact to a registry collection brings that artifact version from a private, project-level scope, to the shared organization level scope.
:::

## How to link an artifact version 

Based on your use case, follow the instructions described in the tabs below to link an artifact version.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">


Programmatically link an artifact version to a collection with [`link_artifact`](../../ref/python/run.md#link_artifact). Before you link an artifact to a collection, ensure that the registry that the collection belongs to already exists.


Use the `target_path` parameter to specify the collection and registry you want to link the artifact version to. The target path consists of:

```text
{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

Copy and paste the code snippet below to link an artifact version to a collection within an existing registry. Replace values enclosed in `<>` with your own:

```python
import wandb

TEAM_ENTITY_NAME = "<team_entity_name>"
ORG_ENTITY_NAME = "<org_entity_name>"

REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"

run = wandb.init(
        entity=TEAM_ENTITY_NAME, project="<project_name>")

artifact = wandb.Artifact(name="<artifact_name>", type="<collection_type>")
artifact.add_file(local_path="<local_path_to_artifact>")

target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
run.link_artifact(artifact = artifact, target_path = target_path)
```

If you want to link an artifact version to the **Models** registry or the **Dataset** registry, set the artifact type to `"model"` or `"dataset"`, respectively.


For example, the proceeding code snippet simulates logging a model artifact called "my_model.txt" to a collection called "Example ML Task" within the model registry:

```python
import wandb

TEAM_ENTITY_NAME = "<team_entity_name>"
ORG_ENTITY_NAME = "<org_entity_name>"

REGISTRY_NAME = "model" 
COLLECTION_NAME = "Example ML Task"

COLLECTION_TYPE = "model"


with wandb.init(entity=TEAM_ENTITY_NAME, project="link_quickstart") as run:
  with open("my_model.txt", "w") as f:
    f.write("simulated model file")

  logged_artifact = run.log_artifact("./my_model.txt", "artifact-name", type=COLLECTION_TYPE)
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
  )
```


  </TabItem>
  <TabItem value="registry_ui">

1. Navigate to the Registry App.
![](/images/registry/navigate_to_registry_app.png)
2. Hover your mouse next to the name of the collection you want to link an artifact version to.
3. Select the meatball menu icon (three horizontal dots) next to  **View details**.
4. From the dropdown, select **Link new version**.
5. From the sidebar that appears, select the name of a team from the **Team** dropdown.
5. From the **Project** dropdown, select the name of the project that contains your artifact. 
6. From the **Artifact** dropdown, select the name of the artifact. 
7. From the **Version** dropdown, select the artifact version you want to link to the collection.

<!-- TO DO insert gif -->

  </TabItem>
  <TabItem value="artifacts_ui">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Click on the artifact version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select an artifact from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 

<!-- Update this gif -->
<!-- ![](/images/models/manual_linking.gif) -->

  </TabItem>
</Tabs>



:::tip Linked vs source artifact versions
* Source version: the artifact version inside a team's project that is logged to a [run](../runs/intro.md).
* Linked version: the artifact version that is published to the registry. This is a pointer to the source artifact, and is the exact same artifact version, just made available in the scope of the registry.
:::

## Troubleshooting 

Below are some common things to double check if you are not able to link an artifact. 

### Logging artifacts from a personal account

Make sure that you log artifacts using a team entity within your organization. Only artifacts logged within an organization's team can be linked to the organization's registry. 

Artifacts logged to W&B with a personal entity can not be linked to the registry.

#### How to log from a team entity
1. Specify the team as the entity when you initialize a run with [`wandb.init()`](https://docs.wandb.ai/ref/python/init). If you do not specify the `entity` when you initialize a run, the run uses your default entity which may or may not be your team entity. 
  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity_name>', 
    project='<project_name>'
    )
  ```
2. Log the artifact to the run:

    ```python
    run.log_artifact(artifact)
    ```
3. If an artifact is logged to your personal entity, you will need to re-log it to an entity within your organization.


### Organization names with team name collisions

W&B appends a unique hash to the organization name to avoid naming collisions with existing entities. The combination of the name and the unique hash is known as an organizational identifier or `ORG_ENTITY_NAME`.

For example, if your organization name is "reviewco" and you also have a team named "reviewco", W&B appends a hash to the organization name that results in an `ORG_ENTITY_NAME` named `reviewco_XYZ123456`. 

:::tip 
When linking to a registry with the Python SDK, always use the ORG_ENTITY_NAME format in the target_path. In this case, the target path takes the form of `{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}`. 
:::

For example, the target path might look like `reviewco_XYZ123456/wandb-registry-model/my-collection`.



### Confirm the path of a registry

To verify the exact path for linking:

1. Create or inspect an empty collection inside a registry.
2. In the details of the collection, look for the `target_path` field. This field shows the `ORG_ENTITY_NAME`.

<!-- insert screenshot -->

