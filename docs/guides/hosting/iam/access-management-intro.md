---
displayed_sidebar: default
---

# Access management

[INSERT - to do]

## Manage users and teams within an organization
The first user to sign up to W&B with a unique organization domain is assigned as that organization's *instance administrator role*. The organization administrator assigns specific users team administrator roles.

:::note
W&B recommends to have more than one instance admin in an organization. It is a best practice to ensure that admin operations can continue when the primary admin is not available. 
:::

A *team administrator* is a user in organization that has administrative permissions within a team. 


The organization administrator can access and use an organization's account settings at `https://wandb.ai/account-settings/` to invite users, assign or update a user's role, create teams, remove users from your organization, assign the billing administrator, and more. See [Add and manage users](./manage-organization.md#add-and-manage-users) for more information. 

Once an organization administrator creates a team, either the instance administrator or team administrator can invite users to that team, assign or update a team member's role, automatically add new users to a team when they join your organization, remove users from a team, and more. Both the organization administrator and the team administrator use team dashboards at `https://wandb.ai/<your-team-name>` to manage teams. For more information on what organization administrators and team administrators can do, see [Add and manage teams](./manage-organization.md#add-and-manage-teams).


## Limit visibility to specific projects

Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it. Limiting who can view a project is particularly useful if a team works with sensitive or confidential data.

An organization admin, team admin, or the owner of a project can both set and edit a project's visibility. 

For more information, see [Project visibility](./restricted-projects.md).


