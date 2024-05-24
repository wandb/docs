---
displayed_sidebar: default
---

# Create a custom registry
Create custom registry within an organization. This is particularly useful for organizing project-specific requirements. For example, you might need a registry dedicated to storing experiment configurations or hyperparameters for a specific research project. 

:::info
W&B Registry is is private preview and in active development. Artifact types cannot be edited once defined.  
:::

1. Navigate to the Registries App in the W&B App UI.
2. Within **Custom registries**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown. See [LINK] for more information on how to limit visibility.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry permits.
8. Click on the **Create registry** button. 

![](/images/registry/create_custom_registry.png)

The preceding image shows a custom registry called Forecast that is about to be created. The registry is set to **Restricted** [LINK] which means that only members that are manually added to the Forecast registry will have access to this registry. Only artifacts with `hf-model` type can be added to the Forecast registry (see the **Specify types** field). 

## Restrict registry visibility
<!-- Who can do this? -->
Restrict who can access a custom W&B Registry with visibility restrictions. A custom registry can have either **Restricted** or **Organization** visibility:

* Organization visibility: Anyone in the organization can view the registry.
* Restricted visibility: Only invited organization members can view and edit the registry.


You can restrict visibility to a registry when you create a registry or after the registry is created. The following steps describe how to restrict the visibility of a custom registry that already exists:

1. Navigate to the Registries App in the W&B App UI.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select your desired registry visibility.
Continue if you select **Restricted visibility** .
5. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
6. Within the **Member** field, add the email or username of the member you want to add.
7. Click **Add new member**.

For more information about registry roles, see [LINK].