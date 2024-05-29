---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

Create a collection within a registry to organize your artifacts. A *collection* is a set of linked artifact versions in a registry. Each collection represents a distinct task or use case and serves as a container for a curated selection of artifact versions related to that task.

For example, the proceeding image shows a registry named "Forecast". Within the "Forecast" registry there are two collections called "LowLightPedRecog-YOLO" and "TextCat". "LowLightPedRecog-YOLO" collection contains artifacts from machine learning experiments that uses the You Only Look Once YOLO detection algorithm.  Whereas the "TextCat" collection contains artifacts from [INSERT]. 

![](/images/registry/what_is_collection.png)

## Create a collection within a core registry

The following steps describe how to create a collection within a core registry.

1. Navigate to the Registries App in the W&B App UI.
2. Select a core registry.
3. Click on the **Create collection** button in the upper right hand corner.
4. Provide a name for your collection in the **Name** field. 
5. Select a type from the **Type** dropdown.
5. Optionally provide a description of your collection in the **Description** field.
6. Optionally add one or more tags in the **Tags** field. 
7. Click **Link version**.
8. From the **Project** dropdown, select the project where your artifact is stored.
9. From the **Artifact** collection dropdown, select your artifact.
10. From the **Version** dropdown, select the artifact version you want to link to your collection.
11. Click on the **Create collection** button.

:::note
Administrators of a registry can optionally add additional artifact types that the registry accepts. Note that you can not remove artifacts types from a core registry once they are added.
:::

## Create a collection within a custom registry

When you [create a custom registry](./create_registry), you can optionally restrict the artifact types can be linked to a given collection. 

1. Navigate to the Registries App in the W&B App UI.
2. Select a custom registry.
3. Click on the **Create collection** button in the upper right hand corner.
4. Provide a name for your collection in the **Name** field. 
5. In the **Type** field, provide one or more artifact types that this collection accepts. 
5. Optionally provide a description of your collection in the **Description** field.
6. Optionally add one or more tags in the **Tags** field. 
7. Click **Link version**.
8. From the **Project** dropdown, select the project where your artifact is stored.
9. From the **Artifact** collection dropdown, select your artifact.
10. From the **Version** dropdown, select the artifact version you want to link to your collection.
11. Click on the **Create collection** button.




