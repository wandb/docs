---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link an artifact version to a registry

Programmatically or interactively link artifact versions to a collection.

:::info
When you link an artifact to a registry, this "publishes" that artifact to that registry. Any user that has access to that registry can access linked artifact versions when you link an artifact to a collection.

In other words, linking an artifact to a collection brings that artifact version from a private, project-level scope, to the shared organization level scope.
:::

## Link an artifact version
Link an artifact version with the W&B Python SDK, Registry App, or with the Artifact browser.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">


Use the [`link_artifact`](../../ref/python/run.md#link_artifact) method to programmatically link an artifact to a registry. When you link an artifact, specify the path where you want artifact version to link to for the `target_path` parameter. The target path takes the form of `"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"`.

Replace values enclosed in `<>` with your own:

```python
import wandb

ARTIFACT_NAME = "<ARTIFACT-TO-LINK>"
ARTIFACT_TYPE = "ARTIFACT-TYPE"
ENTITY_NAME = "<TEAM-ARTIFACT-BELONGS-IN>"
PROJECT_NAME = "<PROJECT-ARTIFACT-TO-LINK-BELONGS-IN>"

ORG_ENTITY_NAME = "<YOUR ORG NAME>"
REGISTRY_NAME = "<REGISTRY-TO-LINK-TO>"
COLLECTION_NAME = "<REGISTRY-COLLECTION-TO-LINK-TO>"

run = wandb.init(entity=ENTITY_NAME, project=PROJECT_NAME)
artifact = wandb.Artifact(name=ARTIFACT_NAME, type=ARTIFACT_TYPE)
run.link_artifact(
    artifact=artifact,
    target_path=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
run.finish()
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

## View the source of linked artifacts

There are two ways to view the source of linked artifacts in a collection: The artifact browser within the project that the artifact is logged to and the Registry App.

A pointer connects a specific artifact version in the registry to the source artifact (located within the project the artifact is logged to). The source artifact also has a pointer to the registry.

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Registry App', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navigate to the Registry App.
2. Select **View details** next the name of the collection where your artifact is linked to.
3. Within the **Versions** section, select **View** next to the artifact version you want to investigate.
4. Click on the **Version** tab within the right panel.
5. Within the **Version overview** section there is a row that contains a **Source Version** field. The **Source Version** field shows both the name of the artifact and the artifacts's version.

For example, the following image shows a `v0` model version called `artifact-joe-clo` (see **Source version** field `artifact-joe-clo:v0`), linked to a collection called `"Joe Clos other collection"`.

![](/images/registry/view_linked_artifact.png)

  </TabItem>
  <TabItem value="browser">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Expand the **model** dropdown menu from the Artifacts panel.
4. Select the name and version of the model linked to the model registry.
5. Click on the **Version** tab within the right panel.
6. Within the **Version overview** section there is a row that contains a **Linked To** field. The **Linked To** field shows both the name of the artifact and the version it possesses(`registered-model-name:version`). 

For example, in the following image, there is a collection called `Joe Clos other collection` (see the **Linked To** field). A model version called `artifact-joe-clo` with a version `v0`(`artifact-joe-clo:v0`) points to the `Joe Clos other collection` registered model.


![](/images/models/view_linked_model_artifacts_browser.png)


  </TabItem>
</Tabs>