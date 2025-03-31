---
cascade:
- url: guides/hosting/iam/access-management/:filename
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-_index
    parent: identity-and-access-management-iam
title: Access management
url: guides/hosting/iam/access-management-intro
weight: 2
---

## Manage users and teams within an organization
The first user to sign up to W&B with a unique organization domain is assigned as that organization's *instance administrator role*. The organization administrator assigns specific users team administrator roles.

{{% alert %}}
W&B recommends to have more than one instance admin in an organization. It is a best practice to ensure that admin operations can continue when the primary admin is not available. 
{{% /alert %}}

A *team administrator* is a user in organization that has administrative permissions within a team. 


The organization administrator can access and use an organization's account settings at `https://wandb.ai/account-settings/` to invite users, assign or update a user's role, create teams, remove users from your organization, assign the billing administrator, and more. See [Add and manage users]({{< relref path="./manage-organization.md#add-and-manage-users" lang="ja" >}}) for more information. 

Once an organization administrator creates a team, the instance administrator or a team administrator can:

- By default, only an admin can invite users to that team or remove users from the team. To change this behavior, refer to [Team settings]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ja" >}}).
- Assign or update a team member's role.
- Automatically add new users to a team when they join your organization.

Both the organization administrator and the team administrator use team dashboards at `https://wandb.ai/<your-team-name>` to manage teams. For more information, and to configure a team's default privacy settings, see [Add and manage teams]({{< relref path="./manage-organization.md#add-and-manage-teams" lang="ja" >}}).

## Limit visibility to specific projects

Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it. Limiting who can view a project is particularly useful if a team works with sensitive or confidential data.

An organization admin, team admin, or the owner of a project can both set and edit a project's visibility. 

For more information, see [Project visibility]({{< relref path="./restricted-projects.md" lang="ja" >}}).