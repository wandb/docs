---
menu:
  default:
    identifier: access-management-intro
    parent: identity-and-access-management-iam
url: guides/hosting/iam/access-management-intro
cascade:
- url: guides/hosting/iam/access-management/:filename
title: Access management
weight: 2
---

## Manage users and teams within an organization
The first user to sign up to W&B with a unique organization domain is assigned as that organization's *instance administrator role*. The organization administrator assigns specific users team administrator roles.

{{% alert %}}
W&B recommends to have more than one instance admin in an organization. It is a best practice to ensure that admin operations can continue when the primary admin is not available. 
{{% /alert %}}

A *team administrator* is a user in organization that has administrative permissions within a team. 

Organization administrators can access and use an organization's account settings at `https://wandb.ai/account-settings/` to invite users, assign or update a user's role, create teams, remove users from your organization, assign the billing administrator, and more. See [Add and manage users]({{< relref "./manage-organization.md#add-and-manage-users" >}}) for more information. 

Once an organization administrator creates a team, the instance administrator or a team administrator can:

- By default, only an admin can invite users to that team or remove users from the team. To change this behavior, refer to [Team settings]({{< relref "/guides/models/app/settings-page/teams.md#privacy" >}}).
- Assign or update a team member's role.
- Automatically add new users to a team when they join your organization.

Both the organization administrator and the team administrator use team dashboards at `https://wandb.ai/<your-team-name>` to manage teams. For more information, and to configure a team's default privacy settings, see [Add and manage teams]({{< relref "./manage-organization.md#add-and-manage-teams" >}}).

## Maintain admin access
You must ensure that at least one admin user exists in your instance or organization at all times. Otherwise, no user will be able to configure or maintain your organization's W&B account.

If users are managed interactively, admin access is required to delete a user, including another admin user. This helps to reduce the risk of the sole admin user being removed.

However, if an organization uses automated processes to deprovision users from W&B, a deprovisioning operation could inadvertently remove the last remaining admin from the instance or organization.

For assistance with developing operational procedures, or to restore admin access, contact [support](mailto:support@wandb.com).

## Limit visibility to specific projects

Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it. Limiting who can view a project is particularly useful if a team works with sensitive or confidential data.

An organization admin, team admin, or the owner of a project can both set and edit a project's visibility. 

For more information, see [Project visibility]({{< relref "./restricted-projects.md" >}}).