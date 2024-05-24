---
displayed_sidebar: default
---

# Create a custom registry
Create a custom registry for project-specific requirements. For example, you might need a registry dedicated to storing experiment configurations or hyperparameters for a specific research project.

When you create a custom registry you can restrict the types of artifacts that can be added to that collection. 

:::info
W&B Registry is is private preview and in active development. Artifact types cannot be edited once defined.  
:::

1. Navigate to the Registries App in the W&B App UI.
2. Within **Custom registries**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry permits.
8. Click on the **Create registry** button.

For example, the in the proceeding image, a user chose to limit the types of artifacts that can be added to the **Forecast** registry. More specifically, only artifacts with `hf-model` types can be added to the Forecast registry. 

![](/images/registry/create_custom_registry.png)