---
menu:
  default:
    identifier: configure_registry
    parent: registry
title: Configure registry access
weight: 3
---

A registry admin can [configure registry roles]({{< relref "configure_registry.md#configure-registry-roles" >}}), [add users]({{< relref "configure_registry.md#add-a-user-or-a-team-to-a-registry" >}}), or [remove users]({{< relref "configure_registry.md#remove-a-user-or-team-from-a-registry" >}}) from a registry by configuring the registry's settings.

## Manage users

### Add a user or a team

Registry admins can add individual users or entire teams to a registry. To add a user or team to a registry:

1. Navigate to the Registry at https://wandb.ai/registry/.
2. Select the registry you want to add a user or team to.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. In the **Registry access** section, click **Add access**.
5. Specify one or more user names, emails, or the team names to the **Include users and teams** field.
6. Click **Add access**.

{{< img src="/images/registry/add_team_registry.gif" alt="Adding teams to registry" >}}

Learn more about [configuring user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}), or [Registry role permissions]({{< relref "configure_registry.md#registry-role-permissions" >}}) . 

### Remove a user or team
A registry admin can remove individual users or entire teams from a registry. To remove a user or team from a registry:

1. Navigate to the Registry at https://wandb.ai/registry/.
2. Select the registry you want to remove a user from.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. Navigate to the **Registry access** section and type in the username, email, or team you want to remove.
5. Click the **Delete** button.

{{% alert %}}
Removing a user from a team also removes that user's access to the registry.
{{% /alert %}}

### Change the owner

A Registry admin can designate any member as a registry's owner, including a **Restricted Viewer** or a **Viewer**. Registry ownership is primarily for accountability purposes and does not confer any additional permissions beyond those granted by the user's assigned role.

To change the owner:
1. Navigate to the Registry at https://wandb.ai/registry/.
2. Select the registry you want to configure.
3. Click the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Hover over the row for a member.
6. Click the **...** action menu at the end of the row, then click **Make owner**.


## Configure Registry roles

This section shows how to configure roles for Registry members. For more information about Registry roles, including the cabilities of each role, order of precedence, defaults, and more, see [Details about Registry roles](#details-about-registry-roles).

1. Navigate to the Registry at https://wandb.ai/registry/.
2. Select the registry you want to configure.
3. Click the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user or team you want to edit permissions for.
6. In the **Registry role** column, click the user's role. 
7. From the dropdown, select the role you want to assign to the user.

## Details about Registry roles

There are two different types of roles in W&B: [Team roles]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) and [Registry roles]({{< relref "configure_registry.md#configure-registry-roles" >}}). Your role in a team has no impact or relationship to your role in any registry.

The following sections give more information about Registry roles.

### Default roles
W&B automatically assigns a default **registry role** to a user or team when they are added to a registry. This role determines what they can do in that registry. 

| Entity | Dedicated Cloud or Self-Managed<br />Default registry role | Multi-tenant Cloud<br />Default registry role |
| ----- | ----- | ----- |
| Team | Viewer | Restricted Viewer |
| Member (non admin) | Viewer | Restricted Viewer |
| Org admin | Admin | Admin |

A registry admin can assign or modify roles for users and teams in a registry.
See [Configure user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}) for more information.

{{% alert title="Restricted Viewer role availability" %}}
The **Restricted Viewer** role is currently available only in Multi-Tenant Cloud organizations by invitation only. To request access, or to express interest in the feature on Dedicated Cloud or Self-Managed, [contact support](mailto:support@wandb.ai).

This role provides read-only access to registry artifacts without the ability to create, update, or delete collections, automations, or other registry resources.

**Key differences from Viewer role:**
- Can view artifact metadata and collection details.
- Cannot download artifact files or access file contents.
- Cannot use artifacts with `use_artifact()` in the W&B SDK.
- Cannot add or remove aliases or tags from artifacts.
- Cannot create, edit, or delete collections, automations, or other registry resources.
{{% /alert %}}

### Role permissions
The following table lists each Registry role, along with the permissions provided by each role:

| Permission                                                     | Permission Group | Restricted Viewer<br />(Multi-tenant Cloud, by invitation) | Viewer | Member | Admin | 
|--------------------------------------------------------------- |------------------|-------------------|--------|--------|-------|
| View a collection's details                                    | Read             |        ✓         |   ✓    |   ✓    |   ✓   |
| View a linked artifact's details                               | Read             |        ✓         |   ✓    |   ✓    |   ✓   |
| Usage: Consume an artifact in a registry with use_artifact     | Read             |                   |   ✓    |   ✓    |   ✓   |
| Download a linked artifact                                     | Read             |                   |   ✓    |   ✓    |   ✓   |
| Download files from an artifact's file viewer                  | Read             |                   |   ✓    |   ✓    |   ✓   |
| Search a registry                                              | Read             |        ✓         |   ✓    |   ✓    |   ✓   |
| View a registry's settings and user list                       | Read             |        ✓         |   ✓    |   ✓    |   ✓   |
| Create a new automation for a collection                       | Create           |                   |        |   ✓    |   ✓   |
| Turn on Slack notifications for new version being added        | Create           |                   |        |   ✓    |   ✓   |
| Create a new collection                                        | Create           |                   |        |   ✓    |   ✓   |
| Create a new custom registry                                   | Create           |                   |        |   ✓    |   ✓   |
| Edit collection card (description)                             | Update           |                   |        |   ✓    |   ✓   |
| Edit linked artifact description                               | Update           |                   |        |   ✓    |   ✓   |
| Add or delete a collection's tag                               | Update           |                   |        |   ✓    |   ✓   |
| Add or delete an alias from a linked artifact                  | Update           |                   |        |   ✓    |   ✓   |
| Link a new artifact                                            | Update           |                   |        |   ✓    |   ✓   |
| Edit allowed types list for a registry                         | Update           |                   |        |   ✓    |   ✓   |
| Edit custom registry name                                      | Update           |                   |        |   ✓    |   ✓   |
| Delete a collection                                            | Delete           |                   |        |   ✓    |   ✓   |
| Delete an automation                                           | Delete           |                   |        |   ✓    |   ✓   |
| Unlink an artifact from a registry                             | Delete           |                   |        |   ✓    |   ✓   |
| Edit accepted artifact types for a registry                    | Admin            |                   |        |        |   ✓   |
| Change registry visibility (Organization or Restricted)        | Admin            |                   |        |        |   ✓   |
| Add users to a registry                                        | Admin            |                   |        |        |   ✓   |
| Assign or change a user's role in a registry                   | Admin            |                   |        |        |   ✓   |


### Inherited permissions and precedence

A user's permission in a registry depends on the highest level of privilege assigned to that user, whether individually or by team membership.

For example, suppose a registry admin adds a user called Nico to Registry A and assigns them a **Viewer** registry role. A registry admin then adds a team called Foundation Model Team to Registry A and assigns Foundation Model Team a **Member** registry role.

Nico is a member of the Foundation Model Team, which is a **Member** of the Registry. Because **Member** has more permission than **Viewer**, W&B grants Nico the **Member** role.

The proceeding table demonstrates the highest level of permission in the event of a conflict between a user's individual registry role and the registry role of a team they are a member of:

| Team registry role | Individual registry role | Inherited registry role |
| ------ | ------ | ------ | 
| Viewer | Viewer | Viewer |
| Viewer | Restricted Viewer | Viewer |
| Member | Viewer | Member |
| Member | Restricted Viewer | Member |
| Admin  | Viewer | Admin  |
| Admin  | Restricted Viewer | Admin  | 

If there is a conflict, W&B displays the highest level of permissions next to the name of the user.

For example, in the proceeding image Alex inherits **Member** role privileges because they are a member of the `smle-reg-team-1` team.

{{< img src="/images/registry/role_conflict.png" alt="Registry role conflict resolution" >}}


### SDK compatibility

{{% alert title="SDK version requirement" %}}
To use the W&B SDK to access artifacts as a **Restricted Viewer**, you must use W&B SDK version 0.19.9 or higher. Otherwise, SDK commands will result in permission errors.
{{% /alert %}}

When a **Restricted Viewer** uses the SDK, certain functions are not available or work differently.

- **`use_artifact()`**: Not available. Results in permission errors.
- **`artifact.download()`**: Not available. Results in permission errors.
- **`artifact.files()`**: Available for viewing artifact directory structure and artifact metadata only.
- **`artifact.file()`**: Available for viewing artifact metadata only.
- **`artifact.get()`**: Available for viewing artifact metadata only.
- **`artifact.get_path()`**: Available for viewing artifact metadata only.
- **`artifact.get_entry()`**: Available for viewing artifact metadata only.
- **`artifact.json_encode()`**: Available for viewing artifact metadata only.
- **`artifact.verify()`**: Available for viewing artifact metadata only.

### Cross-registry permissions

A user can have different roles in different registries. For example, a user can be a **Restricted Viewer** in Registry A but a **Viewer** in Registry B. In this case:

- The same artifact linked to both registries will have different access levels
- In Registry A, the user is a **Restricted Viewer** and cannot download files or use the artifact
- In Registry B, the user is a **Viewer** and can download files and use the artifact
- In other words, access is determined by the registry in which the artifact is accessed
