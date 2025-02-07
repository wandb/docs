---
menu:
  default:
    identifier: configure_registry
    parent: registry
title: Configure registry access
weight: 3
---

Registry admins can [configure registry roles]({{< relref "configure_registry.md#configure-registry-roles" >}}), [add users]({{< relref "#add-a-user-or-team-to-a-registry" >}}), or [remove users]({{< relref "#remove-a-user-or-team-from-a-registry" >}}) from a registry by configuring the registry's settings.

{{% alert %}}
Only registry admins can [configure registry roles]({{< relref "configure_registry.md#configure-registry-roles" >}}), [add users]({{< relref "configure_registry.md#add-a-user-or-a-team-to-a-registry" >}}), or [remove users]({{< relref "configure_registry.md#remove-a-user-or-team-from-a-registry" >}}) from a registry.
{{% /alert %}}


## Add a user or a team to a registry

Registry admins can add individual users or entire teams to a registry. To add a user or team to a registry:

1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select the registry you want to add a user or team to.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. Navigate to the **Registry access** section.
5. Click on the **Add access** button.
6. Specify one or more user names, emails, or the team names to the **Include users and teams** field.
7. Click **Add access**.


See [Registry role permissions]({{< relref "configure_registry.md#registry-role-permissions" >}}) for more information about registry roles. To edit roles, see [Configure user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}).

## Remove a user or team from a registry
Registry admins can remove individual users or entire teams from a registry. To remove a user or team from a registry:

1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select the registry you want to remove a user from.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. Navigate to the **Registry access** section and type in the username, email, or team you want to remove.
5. Click the **Delete** button.



## Registry role permissions

Each user in a registry has a *registry role*, which determines what they can do in that registry. 

W&B automatically assigns default registry roles to a user or team when they are added to a registry. 

| Entity | Default registry role |
| ----- | ----- |
| Team | Viewer |
| User (non admin) | Viewer |
| Org admin | Admin |


Registry admins can assign or modify roles for users and teams in a registry.
See [Configure user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}) for more information.

{{% alert title="W&B role types" %}}
There are two different types of roles in W&B: [Team roles]({{< relref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) and [Registry roles]({{< relref "configure_registry.md#registry-role-permissions" >}}).

Your role in a team has no impact or relationship to your role in any registry.
{{% /alert %}}


The proceeding table lists the different roles a user can have and their permissions:


| Permission                                                     | Permission Group | Viewer | Member | Admin | 
|--------------------------------------------------------------- |------------------|--------|--------|-------|
| View a collection’s details                                    | Read             |   X    |   X    |   X   |
| View a linked artifact’s details                               | Read             |   X    |   X    |   X   |
| Usage: Consume an artifact in a registry with use_artifact     | Read             |   X    |   X    |   X   |
| Download a linked artifact                                     | Read             |   X    |   X    |   X   |
| Download files from an artifact’s file viewer                  | Read             |   X    |   X    |   X   |
| Search a registry                                              | Read             |   X    |   X    |   X   |
| View a registry’s settings and user list                       | Read             |   X    |   X    |   X   |
| Create a new automation for a collection                       | Create           |        |   X    |   X   |
| Turn on Slack notifications for new version being added        | Create           |        |   X    |   X   |
| Create a new collection                                        | Create           |        |   X    |   X   |
| Create a new custom registry                                   | Create           |        |   X    |   X   |
| Edit collection card (description)                             | Update           |        |   X    |   X   |
| Edit linked artifact description                               | Update           |        |   X    |   X   |
| Add or delete a collection’s tag                               | Update           |        |   X    |   X   |
| Add or delete an alias from a linked artifact                  | Update           |        |   X    |   X   |
| Link a new artifact                                            | Update           |        |   X    |   X   |
| Edit allowed types list for a registry                         | Update           |        |   X    |   X   |
| Edit custom registry name                                      | Update           |        |   X    |   X   |
| Delete a collection                                            | Delete           |        |   X    |   X   |
| Delete an automation                                           | Delete           |        |   X    |   X   |
| Unlink an artifact from a registry                             | Delete           |        |   X    |   X   |
| Edit accepted artifact types for a registry                    | Admin            |        |        |   X   |
| Change registry visibility (Organization or Restricted)        | Admin            |        |        |   X   |
| Add users to a registry                                        | Admin            |        |        |   X   |
| Assign or change a user's role in a registry                   | Admin            |        |        |   X   |


### Inherited permissions

A user's permission in a registry depends on the highest level of privilege assigned to that user, whether through a team or directly. 

For example, suppose a registry admin adds a user called Nico to Registry A and assigns them a `Viewer` registry role. The registry admin then adds a team called Team Awesome to Registry A and assigns Team Awesome a `Member` registry role. Nico is a member of Team Awesome.

Since Nico is a member of Team Awesome, W&B grants Nico `Member` registry role permissions because `Member` registry roles have higher level of permissions than `Viewer`. 

The proceeding table shows the highest level of permission in the event of a conflict:

| Registry Role A | Registry role B  | Inherited registry role |
| ------ | ------ | ------ | 
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  | 

W&B displays the highest level of permissions next to the name of the user in the event of a conflict.

<!-- to do: add image -->

## Configure registry roles
1. Navigate to the Registry App at https://wandb.ai/registry/.
2. Select the registry you want to configure.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.


