---
displayed_sidebar: default
---

# Configure registry access
Registries, and the linked artifacts inside of them, are scoped at the organization level. This means that you can share artifacts linked to a registry between teams.

There are two ways to limit who can view and access a registry: user roles for specific registries and registry visibility settings.

* [Define user roles for specific registries](#define-permissions-in-a-registry): Organization admins can choose to set a user's access to viewer or member for each registry. For example, a user with member access to the default Models registry can be restricted to viewer access for the Dataset registry. User-defined roles are useful for fine-grained control over specific members within an organization.
* [Define registry visibility](#restrict-registry-visibility): Restrict who can view a custom registry. When you create a custom registry you can define **Restricted** or **Organization** visibility.

:::tip
The term 'user' is an all encompassing term that refers to any person that uses W&B, independent of their role type.
:::

## Registry roles

A user can have different roles, and therefore access, for each registry in their organization. Organization administrators can set a given user's permission.

The proceeding table lists the different roles a user can have and their permissions:


| Role | Permissions |
| --- | --- |
| Admin | | 
| Member | | 
| Viewer | Read only access. | 
| Owner | | 



## Define permissions in a registry
Restrict who can view a W&B Registry with visibility restrictions. You can restrict visibility to a registry when you create a registry or after the registry is created. 

### Edit a user's role in an existing registry
1. Navigate to the Registries App in the W&B App UI.
2. Select the registry you want to configure a user's role permission.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.

![](/images/registry/edit_registry_role.png)

### Remove a member from a registry
1. Navigate to the Registries App in the W&B App UI.
2. Select a default or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section and type in the username of the member you want to remove.
5. Click on **Delete**.



## Restrict registry visibility
<!-- Who can do this? -->
Restrict who can view a W&B Registry with visibility restrictions. A registry can have either **Restricted** or **Organization** visibility:
* Organization visibility: Anyone in the organization can view the registry.
* Restricted visibility: Only invited organization members can view and edit the registry.
You can restrict visibility to a registry when you create a registry or after the registry is created. 

The following steps describe how to restrict the visibility of a registry:

1. Navigate to the Registries App in the W&B App UI.
2. Select a default or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. From the **Registry visibility** dropdown, select your desired registry visibility.
Continue if you select **Restricted visibility** .
5. Add members of your organization that you want to have access to this registry. Scroll to the **Registry members and roles** section and click on the **Add member** button. 
6. Within the **Member** field, add the email or username of the member you want to add.
7. Click **Add new member**.

For more information about registry roles, see [LINK].


