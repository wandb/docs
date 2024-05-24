---
displayed_sidebar: default
---

# Configure registry access
Registries, and the linked artifacts inside of them, are scoped at the organization level. This means that teams within an organization can share artifacts linked in a registry.

Organization admins can limit who can access a registry by configuring user-defined roles for that registry. User defined roles are useful for fine-grained control on how specific members within an organization can interact with a registry. 

:::tip
The term 'user' is an all encompassing term that refers to any person that uses W&B, independent of their role type.
:::

For example, a user with member access in the core Model registry can be be assigned viewer access in the core Dataset registry.
## Registry roles

A user within an organization can have different roles, and therefore permissions, for each registry in their organization. Organization administrators can set a given user's permission.

The proceeding table lists the different roles a user can have and their permissions:


| Role | Permissions |
| --- | --- |
| Admin | | 
| Member | | 
| Viewer | Read only access. | 
| Owner | | 


## Configure user roles in a registry
1. Navigate to the Registries App in the W&B App UI.
2. Select the registry you want to configure a user's role permission.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.

![](/images/registry/edit_registry_role.png)

## Remove a member from a registry
1. Navigate to the Registries App in the W&B App UI.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section and type in the username of the member you want to remove.
5. Click on **Delete**.






