---
menu:
  default:
    identifier: configure_registry
    parent: registry
title: Configure registry access
weight: 3
---

Add or remove a user or a team to a registry. Assign registry roles to individuals users or teams for greater control of user and team permissions.

## Add a user or a team to a registry

A team administrator or registry administrator can add a user or a team to a registry. To add a user or a team to a registry:

1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select the registry you want to add a user or team to.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry access** section.
5. Click on the **Add access** button.
6. Add a user name, email, or the name of a team to the **Include users and teams** field.
7. Click **Add access**.


W&B assigns all users, including members of a team, a [`Viewer` registry role]({{< relref "configure_registry.md#registry-role-permissions" >}}) when they are added to a registry by default. 

To edit a user's role, see [Configure user roles in a registry]({{< relref "configure_registry.md#configure-user-roles-in-a-registry" >}}).

## Remove a user or team from a registry
1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section and type in the username, email, or team you want to remove.
5. Click the **Delete** button.



## Registry role permissions

Registry administrators can assign each user in an organization a *registry role* to control that user's permissions. Registry roles determine what users can do in a given registry. 

{{% alert %}}
Only registry administrators can [restrict visibility]({{< relref "configure_registry.md#restrict-visibility-to-a-registry" >}}), [configure user roles]({{< relref "configure_registry.md#configure-user-roles-in-a-registry" >}}), or [remove users]({{< relref "configure_registry.md#remove-a-user-from-a-registry" >}}) from registries in an organization.
{{% /alert %}}

A user can have different roles in different registries. For example, a user can have a `Viewer` role in "Registry A" and a `Member` role in the "Registry B".

{{% alert title="W&B role types" %}}
There are two different types of roles in W&B: [Team roles]({{< relref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) and [Registry roles]({{< relref "configure_registry.md#registry-role-permissions" >}}).
{{% /alert %}}

The proceeding table lists the different roles a user can have and their permissions:


| Permission                                                     | Permission Group | Viewer | Member | Admin | Owner |
|--------------------------------------------------------------- |------------------|--------|--------|-------|-------|
| View a collection’s details                                    | Read             |   X    |   X    |   X   |   X   |
| View a linked artifact’s details                               | Read             |   X    |   X    |   X   |   X   |
| Usage: Consume an artifact in a registry with use_artifact     | Read             |   X    |   X    |   X   |   X   |
| Download a linked artifact                                     | Read             |   X    |   X    |   X   |   X   |
| Download files from an artifact’s file viewer                  | Read             |   X    |   X    |   X   |   X   |
| Search a registry                                              | Read             |   X    |   X    |   X   |   X   |
| View a registry’s settings and user list                       | Read             |   X    |   X    |   X   |   X   |
| Create a new automation for a collection                       | Create           |        |   X    |   X   |   X   |
| Turn on Slack notifications for new version being added        | Create           |        |   X    |   X   |   X   |
| Create a new collection                                        | Create           |        |   X    |   X   |   X   |
| Create a new custom registry                                   | Create           |        |   X    |   X   |   X   |
| Edit collection card (description)                             | Update           |        |   X    |   X   |   X   |
| Edit linked artifact description                               | Update           |        |   X    |   X   |   X   |
| Add or delete a collection’s tag                               | Update           |        |   X    |   X   |   X   |
| Add or delete an alias from a linked artifact                  | Update           |        |   X    |   X   |   X   |
| Link a new artifact                                            | Update           |        |   X    |   X   |   X   |
| Edit allowed types list for a registry                         | Update           |        |   X    |   X   |   X   |
| Edit custom registry name                                      | Update           |        |   X    |   X   |   X   |
| Delete a collection                                            | Delete           |        |   X    |   X   |   X   |
| Delete an automation                                           | Delete           |        |   X    |   X   |   X   |
| Unlink an artifact from a registry                             | Delete           |        |   X    |   X   |   X   |
| Edit accepted artifact types for a registry                    | Admin            |        |        |   X   |   X   |
| Change registry visibility (Organization or Restricted)        | Admin            |        |        |   X   |   X   |
| Add users to a registry                                        | Admin            |        |        |   X   |   X   |
| Assign or change a user's role in a registry                   | Admin            |        |        |   X   |   X   |


### Resolving role conflicts

When an administrator adds a user to a registry, W&B attempts to assign that user a `Viewer` registry role by default. In the event of a conflict between a user's [team role]({{< relref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) and default registry role, W&B assigns the user the highest level of privilege as the registry role.

For example, suppose a user has a `Member` team role in "Registry A". A registry administrator then adds that user to "Registry B". That user will have a `Member` role in "Registry B" because it has more privilege than the default `Viewer` role.

The proceeding table shows the inherited registry role in the event of a conflict between a user's team role and default registry role:

| Team role | Registry role | Inherited registry role |
| ------ | ------ | ------ | 
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  |



## Configure user roles in a registry
1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select the registry you want to configure.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.

<!-- To do: add new image -->
<!-- {{< img src="/images/registry/configure_role_registry.gif" alt="" >}} -->

