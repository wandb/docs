---
menu:
  default:
    identifier: configure_registry
    parent: registry
title: Configure registry access
weight: 3
---

<!-- A registry, and the linked artifacts inside a registry, belong to an organization. This means that teams within an organization can publish and consume artifacts linked to a registry, if that team has correct access control. -->

<!-- Registry admins can limit who can access a registry by navigating to a registry's settings and assigning a user's role to [Admin, Member, or Viewer]({{< relref "#registry-roles-permissions" >}}).   -->

Registry administrators can assign each user in an organization a *registry role* to control that user's permissions. Registry roles determine what users can do in a given registry. 

{{% alert %}}
Only registry admins can [restrict visibility]({{< relref "#restrict-visibility-to-a-registry" >}}), [configure user roles]({{< relref "#configure-user-roles-in-a-registry" >}}), or [remove users]({{< relref "#remove-a-user-from-a-registry" >}}) from registries in an organization.
{{% /alert %}}

## Registry role permissions

A user can have different roles in different registries. For example, a user can have a `Viewer` role in "Registry A" and a `Member` role in the "Registry B".

{{% alert title="W&B role types" %}}
There are two different types of roles in W&B: [Team roles]({{< relref "/guides/models/app/settings-page/teams.md#team-roles-and-permissions" >}}) and [Registry roles]({{< relref "#registry-roles-permissions" >}}).

Your role in a team has no impact or relationship on your role in any registry.
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



## Configure user roles in a registry
1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to configure.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user you want to edit permissions for.
6. Click on the user's role within the **Registry role** column. 
7. From the dropdown, select the role you want to assign to the user.

{{< img src="/images/registry/configure_role_registry.gif" alt="" >}}

## Remove a user from a registry
1. Navigate to the **Registry** App in the W&B App UI.
2. Select a core or custom registry.
3. Click on the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section and type in the username of the member you want to remove.
5. Click the **Delete** button.



