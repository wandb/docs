---
displayed_sidebar: default
---

# Mange your organization

Manage W&B users in your organization or team. 

The following guide applies to the administrator (admin) of an organization.

If you are looking to simplify user management in your organization, refer to [Automate user and team management](./automate_iam.md).


## Change the name of your organization

1. Navigate to the organization's dashboard at https://wandb.ai/home.
2. Choose the User menu in the upper right corner of the organization dashboard. From the dropdown, Choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Select the **Change name** button.
5. Within the modal that appears, provide a new name for your organization and select the **Save name** button.

## Add and manage users

### Invite a user

1. Navigate to your W&B Organization's dashboard at: https://wandb.ai/home
2. Select the **Users** tab.
3. Choose **Invite new user**.
4. In the modal that appears, provide the email or username of the user in the Email or username field.
5. (Optional but recommended) Add the user to a team from the **Choose team(s)** dropdown menu. You can alternatively [automatically assign a user to a team if their domain matches](#automatically-add-new-users-to-a-team). 
6. From the **Select role** dropdown, select the role to assign to the user. You can change the user's role at a later time. See the table listed in [Assign a role](#assign-a-role) for more information about possible roles.
7. Choose the **Send invite** button.

An invite link is sent to the user's email after you select the **Send invite** button. Once a user accepts the invite, that user can access your W&B organization.


W&B uses a third-party email server to send user invites. W&B provides an option to configure an internal SMTP server if you have a self-managed Dedicated cloud or Self-managed instance and your organization firewall rules restrict sending traffic outside the corporate network. Refer to [these instructions](../smtp.md) to setup the SMTP server.

:::info
The **Add user** option might be not be available if there are no more seats in the license. Reach out to your W&B team if you have difficulty adding users. 
:::


### Assign or update a user's role

W&B assigns an Admin role to new users within an organization by default. You can assign a user's role when you invite them to an organization or and you can modify an existing user's role after they join your organization.

A user within an organization can have one of the proceeding roles:

| Role | Descriptions |
| ----- | ----- |
| Admin | A instance admin who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends more than one admin for an enterprise Dedicated cloud or Self-managed instances. |
| Member | A regular user of the organization, invited by an instance admin. A organization user cannot invite other users or manage existing users in the organization. `Team admins` could add specific organization users to their respective teams (team-level roles described below in **Team roles**). |
| Viewer | A view-only user of your organization, invited by an instance admin. A viewer only has read access to the organization and the underlying teams that they are a part of.  |


### Remove a user

1. Navigate to your W&B Organization's dashboard at: https://wandb.ai/home .
2. Choose the User menu in the upper right corner of the organization dashboard. From the dropdown, Choose **Settings**.
3. Select the **Users** tab.
4. Search for the user you want to remove in the search bar.
5. Select the ellipses or three dots icon (**...**) when it appears.
6. From the dropdown, choose **Remove member**.

### Assign the billing admin
1. Navigate to your W&B Organization's dashboard at: https://wandb.ai/home .
2. Choose the User menu in the upper right corner of the organization dashboard. From the dropdown, Choose **Settings**.
3. Select the **Users** tab.
4. Search for the user you want to remove in the search bar.
5. Under the **Billing admin** column, choose the user you want to assign as the Billing admin.


## Add and manage teams
Use a team's home page as the central hub to explore projects, reports, and runs. Within the team home page there is a **Settings** tab where you can add and manage users, set a team avatar, adjust privacy settings, set up alerts, track usage, and more. 


:::info
Team admins can add and remove users in their teams. A non-admin user in a team can not invite other users to that team, unless team admin enables relevant team settings.

See [**Team roles**](#team-roles) for more information on available roles at the team level.
:::

If you're looking to simplify team management in your organization, refer to [Automate user and team management](./automate_iam.md).



### Create a team
1. Navigate to your W&B Organization's dashboard at: https://wandb.ai/home
2. Select **Create a team to collaborate** on the left navigation panel underneath **Teams**.
![](/images/hosting/create_new_team.png)
3. A modal will appear. Prove a name for your team in the **Team name** field. 
4. Select a storage type. 
5. Choose on the **Create team** button.

This will redirect you to a newly created Team home page at `https://wandb.ai/<team-name>` where `<team-name>` consists of the name you provide when you create a team.

### Invite users to a team

Invite users to a team with the W&B App UI. Members of a team inherit the organization that the team is a part of.

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the global navigation on the left side of the dashboard.
3. Select the **Users** tab.
4. Choose on **Invite a new user**.
5. Within the modal that appears, provide the email of the user in the **Email or username** field and select the role to assign to that user from the **Select a team** role dropdown. For more information about roles a user can have in a team, see [Team roles](#team-roles).
6. Choose on the **Send invite** button.


### Team roles
The proceeding table lists the roles you can assign a member to when they join a team:

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

1. Navigate to the organization's dashboard at https://wandb.ai/home.
2. Choose the User menu in the upper right corner of the organization dashboard. From the dropdown, Choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Choose the **Claim domain** button within **Domain capture**.
5. Provide the email domain in the **Email domain** field.
6. Select the team that you want new users to automatically join from the **Default team** dropdown.
7. Choose the **Claim email** domain button.


A user that joins W&B with the same domain as your organization is automatically added to the team you specify in the preceding steps.


### Remove users from a team
Remove a user from a team within your organization. W&B preserves runs created in a team even if the user is no longer on the team.

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the left navigation bar.
3. Select the **Users** tab.
4. Hover your mouse next to the ame of the user you want to delete. Select the ellipses or three dots icon (**...**) when it appears. 
5. From the dropdown, select **Remove user**. 

### Manage team storage

## Create and assign custom roles

## Privacy