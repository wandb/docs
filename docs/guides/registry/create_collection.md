---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

Create a collection to organize your machine learning models. A *collection* is a set of linked artifact versions in a registry. Each collection represents a distinct task or use case and serves as a container for a curated selection of artifact versions related to that task.

For example, the proceeding image shows a registry name **Forecast**. Within the Forecast registry there are two collections called **LowLightPedRecog-YOLO** and **TextCat**.

![](/images/registry/what_is_collection.png)

Depending on the registry settings configured by your team admin, you may or may not be able to configure artifact types accepted for a given collection. For more information, see [LINK].

<Tabs
  defaultValue="alltypes"
  values={[
    {label: 'Unrestricted collection types', value: 'alltypes'},
    {label: 'Restricted collection types', value: 'restricted'},
  ]}>
  <TabItem value="alltypes">

Create a collection in the W&B Registry that accepts multiple artifact types:

1. Navigate to the Registries App in the W&B App UI.
2. Select a default or custom registry.
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


  </TabItem>
  <TabItem value="restricted">


1. Navigate to the Registries App in the W&B App UI.
2. Select a default or custom registry.
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


  </TabItem>
</Tabs>