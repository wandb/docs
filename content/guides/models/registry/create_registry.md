---
menu:
  default:
    identifier: create_registry
    parent: registry
title: Create a custom registry
weight: 2
---

Use custom registries to organize artifacts that are specific to your organization or project. 
<!-- To do: add an example custom registry -->



<!-- Unlike a [core registry]({{< relref "./registry_types.md#core-registry" >}}), a team administrator can [restrict the visibility]({{< relref "./create_registry.md#assign-the-visibility-of-a-custom-registry" >}}) of a custom registry. Restricting the visibility of a custom registry helps ensure that you can limit who can access a registry.  -->

Team administrators can limit who in an organization can access a custom registry by [restricting the visibility]({{< relref "./create_registry.md#assign-the-visibility-of-a-custom-registry" >}}) of a custom registry. Restricting the visibility of a custom registry helps ensure that only members that you add to a custom registry can access that registry.

{{% pageinfo color="info" %}}
A core registry can only have Organization visibility.
{{% /pageinfo %}}


## Create a custom registry

The following procedure describes how to create a custom registry:
1. Navigate to the **Registry** App at https://wandb.ai/registry/.
2. Within **Custom registry**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown. See [Registry visibility types]({{< relref "./configure_registry.md#registry-visibility-types" >}}) for more information on registry visibility options.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry accepts.
8. Click on the **Create registry** button. 

{{% alert %}}
You can not remove an artifact type from a registry once you add and save it in the registry's settings.
{{% /alert %}}

{{< img src="/images/registry/create_registry.gif" alt="" >}}

For example, the preceding image shows a custom registry called "Fine_Tuned_Models" that a user is about to create. The registry is set to **Restricted** which means that only members that are manually added to the "Fine_Tuned_Models" registry will have access to this registry.


## Registry visibility types

The visibility type of a registry determines who can access the registry. A custom registry can have either a *Restricted* or *Organization* visibility. 

| Visibility | Description | Default role | 
| --- | --- | --- | 
| Organization | Everyone in the org can access the registry. | By default, organization administrators are an admin for the registry. All other users are a viewer in the registry by default. | 
| Restricted   | Only invited organization members can access the registry.| The user who created the restricted registry is the only user in the registry by default, and is the organization's owner. |

## Assign the visibility of a custom registry

A team administrator can assign the visibility of a custom registry during or after the creation of a custom registry. 

To restrict the visibility of an existing custom registry:

1. Navigate to the **Registry** App at https://wandb.ai/registry/.
2. Select a registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select the desired registry visibility.

Continue if you select **Restricted** visibility:

<!-- To do: Add updated process of adding a team -->

5. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
6. Within the **Member** field, add the email or username of the member you want to add.
7. Click **Add new member**.

{{< img src="/images/registry/change_registry_visibility.gif" alt="" >}}

See [Create a custom registry]({{< relref "./create_registry.md#create-a-custom-registry" >}}) for more information on how assign the visibility of a custom registry when a team administrator creates it.