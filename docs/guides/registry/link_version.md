---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link an artifact version to a registry

Programmatically or interactively link artifact versions to a registry.

:::info
When you link an artifact to a registry, this "publishes" that artifact to that registry. Any user that has access to that registry can access linked artifact versions when you link an artifact to a collection.

In other words, linking an artifact to a registry collection brings that artifact version from a private, project-level scope, to the shared organization level scope.
:::

Based on your use case, follow the instructions described in the tabs below to link an artifact version.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">


Use the [`link_artifact`](../../ref/python/run.md#link_artifact) method to programmatically link an artifact to a registry. When you link an artifact, specify the path where you want artifact version to link to for the `target_path` parameter. The target path takes the form of `{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}`. Note that this path informs the registry and collection the artifact will be linked to. 

Replace values enclosed in `<>` with your own:
```python
import wandb

TEAM_ENTITY = "<team-entity>"
ORG_NAME = "<insert-org-name>"
REGISTRY_NAME = "<insert-registry-name>"  # Set to "model" to link to the model registry
COLLECTION_TYPE = "model"

with wandb.init(entity="TEAM_ENTITY", project="link-quickstart") as run:
  with open("my_model.txt", "w") as f:
    f.write("simulated model file")

  logged_artifact = run.log_artifact("./my_model.txt", "artifact-name", type=COLLECTION_TYPE)
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"{ORG_NAME}/wandb-registry-{REGISTRY_NAME}/Example ML Task"
  )
```

If you want to link an artifact version to the **Models** registry or the **Dataset** registry, set the artifact type to `"model"` or `"dataset"`, respectively.

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

Only artifacts logged within an organization's team can be linked to the organization's registry. Make sure that you log artifacts using a team entity within your organization. 

#### How to log from a team entity
1. Specify the team as the entity when you initialize a run with [`wandb.init()`](https://docs.wandb.ai/ref/python/init). If you do not specify the `entity` when you initialize a run, the run uses your default entity which may or may not be your team entity. 
  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```
2. Log the artifact to the run:

    ```python
    run.log_artifact(artifact)
    ```
3. If an artifact is logged to the wrong entity, you will need to re-log it to an entity within your organization.


### Organization names with team name collisions

W&B appends a unique hash to the organization name to avoid naming collisions when you have an organization with a team name that exactly matches the organization name. The combination of the name and the unique hash is known as an organizational identifier or `ORG_IDENTIFIER`.

For example, if your organization name is "reviewco" and you also have a team named "reviewco", W&B appends a hash to the organization name that results in an `ORG-IDENTIFIER` named `reviewco_XYZ123456`. 

:::tip 
When linking to a registry with the Python SDK, always use the ORG_IDENTIFIER format in the target_path. In this case, the target path takes the form of `{ORG_IDENTIFIER}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}`. 
:::

For example, the target path might look like `reviewco_XYZ123456/wandb-registry-model/my-collection`.



### Confirm the path of a registry

To verify the exact path for linking:
1. Check Out an Empty Collection: Create or inspect the empty state for an empty collection inside a registry.
2. Locate the code snippet for link_artifact: In the details of the collection, look for the target_path field. This field will show the ORG_IDENTIFIER.

<!-- insert screenshot -->

