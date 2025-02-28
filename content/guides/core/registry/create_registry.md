---
menu:
  default:
    identifier: create_registry
    parent: registry
title: Create a custom registry
weight: 2
---

A custom registry offers flexibility and control over the artifact types that you can use, allows you to restrict the registry's visibility, and more.

{{% pageinfo color="info" %}}
See the summary table in [Registry types]({{< relref "registry_types.md#summary" >}}) for a complete comparison of core and custom registries.
{{% /pageinfo %}}


## Create a custom registry

To create a custom registry:
1. Navigate to the **Registry** App at https://wandb.ai/registry/.
2. Within **Custom registry**, click on the **Create registry** button.
3. Provide a name for your registry in the **Name** field.
4. Optionally provide a description about the registry.
5. Select who can view the registry from the **Registry visibility** dropdown. See [Registry visibility types]({{< relref "./configure_registry.md#registry-visibility-types" >}}) for more information on registry visibility options.
6. Select either **All types** or **Specify types** from the **Accepted artifacts type** dropdown.
7. (If you select **Specify types**) Add one or more artifact types that your registry accepts.
8. Click on the **Create registry** button. 

{{% alert %}}
An artifact type cannot be removed from a registry once it is saved in the registry's settings.
{{% /alert %}}

For example, the proceeding image shows a custom registry called `Fine_Tuned_Models` that a user is about to create. The registry is **Restricted** to only members that are manually added to the registry.

{{< img src="/images/registry/create_registry.gif" alt="" >}}

## Visibility types

The *visibility* of a registry determines who can access that registry. Restricting the visibility of a custom registry helps ensure that only specified members can access that registry.

There are two type registry visibility options for a custom registry: 

| Visibility | Description |
| --- | --- | 
| Restricted   | Only invited organization members can access the registry.| 
| Organization | Everyone in the org can access the registry. |

A team administrator or registry administrator can set the visibility of a custom registry.

The user who creates a custom registry with Restricted visibility is added to the registry automatically as its registry admin. 


## Configure the visibility of a custom registry

A team administrator or registry administrator can assign the visibility of a custom registry during or after the creation of a custom registry. 

To restrict the visibility of an existing custom registry:

1. Navigate to the **Registry** App at https://wandb.ai/registry/.
2. Select a registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select the desired registry visibility.
5. if you select **Restricted visibility**:
   1. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
   2. Within the **Member** field, add the email or username of the member you want to add.
   3. Click **Add new member**.

{{< img src="/images/registry/change_registry_visibility.gif" alt="" >}}

See [Create a custom registry]({{< relref "./create_registry.md#create-a-custom-registry" >}}) for more information on how assign the visibility of a custom registry when a team administrator creates it.

