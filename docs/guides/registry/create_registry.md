---
displayed_sidebar: default
---

# Create a custom registry
Create a custom registry within an organization. Custom registries are particularly useful for organizing project-specific requirements that differ from the default, core registries that W&B creates by default. 

For example, you might want to create a custom registry for evaluating or fine-tuning datasets.

:::caution
W&B Registries is in active development. Currently, you can not remove an artifact type once they are added and saved.
:::

1. Navigate to the Registries App in the W&B App UI.
2. Within **Custom registries**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown. See [LINK] for more information on registry visibility options.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry permits.
8. Click on the **Create registry** button. 

![](/images/registry/create_custom_registry.png)

The preceding image shows a custom registry called "Forecast" that a user is about to create. The registry is set to **Restricted** [LINK] which means that only members that are manually added to the "Forecast" registry will have access to this registry. In addition, only  `hf-model` artifact types can be added to the "Forecast" registry (see the **Specify types** field). 

## Restrict visibility to a registry
<!-- Who can do this? -->
Restrict who can view and access a custom registry. You can restrict visibility to a registry when you create a custom registry or after you create a custom registry. A custom registry can have either restricted or organization visibility. For more information on registry visibilities, see [LINK].

<!-- | Visibility | Description |
| --- | --- |
| Organization | Anyone in the organization can view the registry. |
| Restricted   | Only invited organization members can view and edit the registry.|  -->

The following steps describe how to restrict the visibility of a custom registry that already exists:

1. Navigate to the Registries App in the W&B App UI.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select your desired registry visibility.
Continue if you select **Restricted visibility** .
5. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
6. Within the **Member** field, add the email or username of the member you want to add.
7. Click **Add new member**.

For more information about registry roles, see [LINK].