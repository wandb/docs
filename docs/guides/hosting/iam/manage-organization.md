---
displayed_sidebar: default
---

# Mange your organization

As an administrator of an organization you can both [manage individual users](#add-and-manage-users) within your organization and you can [manage teams](#add-and-manage-teams). 

As a team administrator you can [manage teams](#add-and-manage-teams).

:::info
The following workflow applies to:
* W&B Multi-tenant SaaS Cloud
* Users with instance administrator (admin) roles. Reach out to an administrator in your organization if you believe you should have instance admin permissions. 
:::

If you are looking to simplify user management in your organization in Dedicated Cloud deployment, refer to [Automate user and team management](./automate_iam.md).


<!-- W&B assigns an Admin role to new users within an organization by default.  -->

## Change the name of your organization

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
3. Within the **Settings** tab, select **General**.
4. Select the **Change name** button.
5. Within the modal that appears, provide a new name for your organization and select the **Save name** button.

## Add and manage users

Use your organization's dashboard to invite users, assign or update a user's role, remove users from your organization, assign the billing administrator, and more.

### Invite a user

1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Users**.
3. Select **Invite new user**.
4. In the modal that appears, provide the email or username of the user in the Email or username field.
5. (Optional but recommended) Add the user to a team from the **Choose team(s)** dropdown menu. You can alternatively [automatically assign a user to a team if their domain matches](#automatically-add-new-users-to-a-team). 
6. From the **Select role** dropdown, select the role to assign to the user. You can change the user's role at a later time. See the table listed in [Assign a role](#assign-a-role) for more information about possible roles.
7. Choose the **Send invite** button.

An invite link is sent using a third-party email server to the user's email after you select the **Send invite** button. A user can access your organization once they accept the invite.

:::note
The **Invite new user** button is active only when your license has available seats. Reach out to your W&B team if you have difficulty adding users. 
:::


:::tip enable SSO for authentication
W&B strongly recommends and encourages that users authenticate to an organization using Single Sign-On (SSO). 

To learn more about how to setup SSO with Dedicated cloud or Self-managed instances, refer to [SSO with OIDC](./sso.md) or [SSO with LDAP](./ldap.md).

Reach out to your W&B for further assistance.
:::

### Auto provision users

A person in your company (someone who has the same domain as your company) can sign in to your W&B Organization with Single Sign-On (SSO) if SSO is set up and the SSO provider permits it.

:::tip 
Auto provisioning with SSO is useful for adding users to an organization at scale because organization admins do not need to generate individual user invitations.
:::

[INSERT - What role do they get?]


Auto-provisioning users with SSO is on by default for Dedicated cloud instances and Self-managed deployments. You can turn off auto provisioning. Turning auto provisioning off enables you to selectively add specific users to your W&B organization.

Reach out to your W&B team if you are on Dedicated Cloud instance and you want to turn off auto provisioning with SSO.

For Self-managed deployments, you can configure configure the setting `DISABLE_SSO_PROVISIONING=true` to turn off auto provisioning with SSO.



### Assign or update a user's role

You initially assign a role to a user when you invite them to your organization. You can change any user's role at a later time.

:::info
The role a user has within an organization does not impact the role that that user can have in a team.
:::

A user within an organization can have one of the proceeding roles:

| Role | Descriptions |
| ----- | ----- |
| Admin | A instance admin who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends more than one admin for an enterprise Dedicated cloud or Self-managed instances. |
| Member | A regular user of the organization, invited by an instance admin. A organization user cannot invite other users or manage existing users in the organization. |
| Viewer | A view-only user of your organization, invited by an instance admin. A viewer only has read access to the organization and the underlying teams that they are a part of.  |

To change a user's role:

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
4. Select a role from the **TEAM ROLE** dropdown next to the name of the user.


### Remove a user

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
5. Select the ellipses or three dots icon (**...**) when it appears.
6. From the dropdown, choose **Remove member**.

### Assign the billing admin
1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
5. Under the **Billing admin** column, choose the user you want to assign as the billing administrator.


## Add and manage teams
Use your organization's dashboard to create teams within your organization. Once an organization admin creates a team, either the org admin or team admin can invite users to that tem, assign or update a team member's role, automatically add new users to a team when they join your organization, remove users from a team, and manage team storage with the team's dashboard at `https://wandb.ai/<team-name>`.

<!-- If you're looking to simplify team management in your organization, refer to [Automate user and team management](./automate_iam.md). -->


### Create a team

Use your organization's dashboard to create a team:

1. Navigate to https://wandb.ai/home.
2. Select **Create a team to collaborate** on the left navigation panel underneath **Teams**.
![](/images/hosting/create_new_team.png)
3. Provide a name for your team in the **Team name** field in the modal that appears. 
4. Choose a storage type. 
5. Select the **Create team** button.

After you select **Create team** button, W&B redirects you to a new team page at `https://wandb.ai/<team-name>`. Where `<team-name>` consists of the name you provide when you create a team.

Once you have a team, you can add users to that team.

### Invite users to a team

Invite users to a team in your organization. Use the team's dashboard to invite users using their email address or W&B username if they already have a W&B account.

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the global navigation on the left side of the dashboard.
![](/images/hosting/team_settings.png)
3. Select the **Users** tab.
4. Choose on **Invite a new user**.
5. Within the modal that appears, provide the email of the user in the **Email or username** field and select the role to assign to that user from the **Select a team** role dropdown. For more information about roles a user can have in a team, see [Team roles](#assign-or-update-a-team-members-role).
6. Choose on the **Send invite** button.

In addition to inviting users manually with email invites, you can automatically add new users to a team if the new user's[ email matches the domain of your organization](#automatically-add-new-users-to-a-team).

### Assign or update a team member's role
The proceeding table lists the roles you can assign to a member of a team:

| Role   |   Definition   |
|-----------|---------------------------|
| Admin     | A user who can add and remove other users in the team, change user roles, and configure team settings.   |
| Member    | A regular user of a team, invited by email or their organization-level username by the team admin. A member user cannot invite other users to the team.  |
| View-Only (Enterprise-only feature) | A view-only user of a team, invited by email or their organization-level username by the team admin. A view-only user only has read access to the team and its contents.  |
| Service (Enterprise-only feature)   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure to set the environment variable `WANDB_USERNAME`  to correctly attribute runs to the appropriate user. |
| Custom Roles (Enterprise-only feature)   | Custom roles allow organization admins to compose new roles by inheriting from the above View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams. Refer to [this article](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details. |

<!-- :::note
W&B recommends to have more than one admin in a team. It is a best practice to ensure that admin operations can continue when the primary admin is not available.

Refer to [Team Service Account Behavior](../../app/features/teams.md#team-service-account-behavior) for more information.
::: -->


:::note
Only enterprise licenses on Dedicated Cloud or Self-managed deployment can assign custom roles to members in a team.
:::

### Automatically add new users to a team

Automatically add new users to a team if the user's domain matches the domain of your organization. 

<!-- Automatically assign new users that join an organization to one or more teams if the user's domain matches the organization's domain. -->

Assigning a team to a user when they onboard helps ensure that that new user does not create assets outside of their organization's account. Assets a user creates outside of an organization is not transferred if that user joins the organization at a later date.


:::info Existing users
Existing users with verified email addresses that match your organization's domain can join your teams within your organization.

Data that a user creates before that user joins an organization is preserved. Note that W&B does not migrate any assets a user creates outside of an organization.
:::

Before you can automatically assign new users to a specific team, you must enable domain matching within that team's settings: 

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`. Where `<team-name>` is the name of the team you want to enable domain matching.
2. Select **Team settings** in the global navigation on the left side of the team's dashboard.
3. Within the **Privacy** section, toggle the "Recommend new users with matching email domains join this team upon signing up" option.


Once you enable domain matching for a team, you can now automatically assign new users to that team when they join your organization:

:::note Domains must be unique
Domains are unique identifiers. This means that you can not use a domain that is already in use by another organization. 
:::

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Choose the **Claim domain** button within **Domain capture**.
5. Provide the email domain in the **Email domain** field.
6. Select the team that you want new users to automatically join from the **Default team** dropdown.
7. Choose the **Claim email** domain button.


A user that joins W&B with the same domain as your organization is automatically added to the team you specify in the preceding steps.


### Remove users from a team
Remove a user from a team using the team's dashboard. W&B preserves runs created in a team even if the member who created the runs is no longer on that team.

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the left navigation bar.
3. Select the **Users** tab.
4. Hover your mouse next to the ame of the user you want to delete. Select the ellipses or three dots icon (**...**) when it appears. 
5. From the dropdown, select **Remove user**. 


<!-- To do as a follow up -->
<!-- ### Manage team storage

## Create and assign custom roles

## Privacy -->