---
displayed_sidebar: default
---

# Create a custom registry
Create a custom registry within an organization. A custom registry is particularly useful for organizing project-specific requirements that differ from the default, core registry that W&B creates by default. 

For example, you might want to create a custom registry for evaluating or fine-tuning datasets.

1. Navigate to the Registry App in the W&B App UI.
2. Within **Custom registry**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown. See [LINK] for more information on registry visibility options.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry accepts.
:::info
An artifact type can not be removed from a registry once it is added and saved in the registry's settings.
:::
8. Click on the **Create registry** button. 

<!-- ![](/images/registry/create_custom_registry.png) -->

![](/images/registry/create_registry.gif)

For exampe, the preceding image shows a custom registry called "Forecast" that a user is about to create. The registry is set to **Restricted** which means that only members that are manually added to the "Forecast" registry will have access to this registry. In addition, only  `hf-model` artifact types can be added to the "Forecast" registry (see the **Specify types** field). 

