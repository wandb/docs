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

1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to add a user or team to.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. In the **Registry access** section, click **Add access**.
5. Specify one or more user names, emails, or the team names to the **Include users and teams** field.
6. Click **Add access**.

{{< img src="/images/registry/add_team_registry.gif" alt="Adding teams to registry" >}}

Learn more about [configuring user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}), or [Registry role permissions]({{< relref "configure_registry.md#registry-role-permissions" >}}) . 

### Remove a user or team
A registry admin can remove individual users or entire teams from a registry. To remove a user or team from a registry:

1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to remove a user from.
3. Click on the gear icon on the upper right hand corner to access the registry settings.
4. Navigate to the **Registry access** section and type in the username, email, or team you want to remove.
5. Click the **Delete** button.

{{% alert %}}
Removing a user from a team also removes that user's access to the registry.
{{% /alert %}}

### Change the owner of a registry

A registry admin can designate any member as a registry's owner, including a **Restricted Viewer** or a **Viewer**. Registry ownership is primarily for accountability purposes and does not confer any additional permissions beyond those granted by the user's assigned role.

To change the owner:
1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to configure.
3. Click the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Hover over the row for a member.
6. Click the **...** action menu at the end of the row, then click **Make owner**.


## Configure Registry roles

This section shows how to configure roles for Registry members. For more information about Registry roles, including the cabilities of each role, order of precedence, defaults, and more, see [Details about Registry roles](#details-about-registry-roles).

1. Navigate to the **Registry** App in the W&B App UI.
2. Select the registry you want to configure.
3. Click the gear icon on the upper right hand corner.
4. Scroll to the **Registry members and roles** section.
5. Within the **Member** field, search for the user or team you want to edit permissions for.
6. In the **Registry role** column, click the user's role. 
7. From the dropdown, select the role you want to assign to the user.

## Details about Registry roles

The following sections give more information about Registry roles.

{{% alert %}}
Your [role in a team]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) has no impact or relationship to your role in any registry.
{{% /alert %}}

### Default roles
W&B automatically assigns a default **registry role** to a user or team when they are added to a registry. This role determines what they can do in that registry. 

| Entity                                 | Default registry role<br />(Dedicated Cloud / Self-Managed) | Default registry role<br />(Multi-tenant Cloud) |
|----------------------------------------|-------------------------------------------------------------|-------------------------------------------------|
| Team                                   | Viewer                                                      | Restricted Viewer                               |
| User or service account (non admin)    | Viewer                                                      | Restricted Viewer                               |
| Service account (non admin)            | Member<sup><a href="#service_account_footnote">1</a></sup>  | Member<sup><a href="#service_account_footnote">1</a></sup> |
| Org admin                              | Admin                                                       | Admin                                           |

<a id="service_account_footnote">1</a>: Service accounts cannot have **Viewer** or **Restricted Viewer** roles.

A registry admin can assign or modify roles for users and teams in the registry.
See [Configure user roles in a registry]({{< relref "configure_registry.md#configure-registry-roles" >}}) for more information.

{{% alert title="Restricted Viewer role availability" %}}
The **Restricted Viewer** role is currently available only in Multi-Tenant Cloud organizations by invitation only. To request access, or to express interest in the feature on Dedicated Cloud or Self-Managed, [contact support](mailto:support@wandb.ai).

This role provides read-only access to registry artifacts without the ability to create, update, or delete collections, automations, or other registry resources.

Unlike a **Viewer**, a **Restricted Viewer**:
- Cannot download artifact files or access file contents.
- Cannot use artifacts with use_artifact() in the W&B SDK.
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


### Inherited Registry role
The registry's membership list shows each user's inherited (effective) registry role (in light gray) next to the role dropdown in their row.

{{< img src="/images/registry/role_conflict.png" alt="Registry membership list showing the user's effective registry role" >}}

A user's effective role in a particular registry matches their _highest_ role among their role in the organization, the registry, and the team that owns the registry, whether inherited or explicitly assigned. For example:

- A team **Admin** or organization **Admin** with the **Viewer** role in a particular registry owned by the team is effectively an **Admin** of the registry.
- A registry **Viewer** with the **Member** role in the team is effectively a **Member** of the registry.
- A team **Viewer** with the **Member** role in a particular registry is effectively a **Member** of the registry.

### SDK compatibility

{{% alert title="SDK version requirement" %}}
To use the W&B SDK to access artifacts as a **Restricted Viewer**, you must use W&B SDK version 0.19.9 or higher. Otherwise, some SDK commands will result in permission errors.
{{% /alert %}}

When a **Restricted Viewer** uses the SDK, certain functions are not available or work differently.

The following methods are not available and result in permission errors:
- [`Run.use_artifact()`]({{< relref "/ref/python/sdk/run/#method-runuse_artifact" >}})
- [`Artifact.download()`]({{< relref "/ref/python/sdk/artifact/#method-artifactdownload" >}})
- [`Artifact.file()`]({{< relref "/ref/python/sdk/artifact/#method-artifactfile" >}})
- [`Artifact.files()`]({{< relref "/ref/python/sdk/artifact/#method-artifactfiles" >}})

The following methods are limited to artifact metadata:
- [`Artifact.get_entry()`]({{< relref "/ref/python/sdk/artifact/#method-artifactget_entry" >}})
- [`Artifact.get_path()`]({{< relref "/ref/python/sdk/artifact/#method-artifactget_path" >}})
- [`Artifact.get()`]({{< relref "/ref/python/sdk/artifact/#method-artifactget" >}})
- [`Artifact.verify()`]({{< relref "/ref/python/sdk/artifact/#method-artifactverify" >}})

### Cross-registry permissions

A user can have different roles in different registries. For example, a user can be a **Restricted Viewer** in Registry A but a **Viewer** in Registry B. In this case:

- The same artifact linked to both registries will have different access levels
- In Registry A, the user is a **Restricted Viewer** and cannot download files or use the artifact
- In Registry B, the user is a **Viewer** and can download files and use the artifact
- In other words, access is determined by the registry in which the artifact is accessed
