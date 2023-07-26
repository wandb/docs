---
displayed_sidebar: default
---
# Manage users
Manage W&B users in your organization or team.

:::info
Users are classified as either an _admin_ or _member_. Admins can add and remove other admins or members.
:::

W&B strongly recommends and encourages user management with Single Sign-On (SSO). To learn more about how to setup SSO with W&B Server, refer to the [SSO Configuration documentation](./sso.md).



## Instance Admins

The first user to sign up to W&B, after you have deployed the W&B Server, is automatically assigned admin permissions. The admin can then add additional users to the instance and create teams.
## Manage your organization
[INSERT intro blurb]

When you invite a user to an organization, you can assign them one of the following roles:

| Role      | Definition                                                                                                                                                                                                                                                                                       |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Admin     | A team member who can add and remove other admins and members of the team.                                                                                                                                                                                                                       |
| Member    | A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.                                                                                                                                                                        |
| View      | A view-only member of your team, invited by email by the team admin. A view-only member only has read access to the team and its contents.                                                                                                                                                       |

### Invite users

1. Navigate to the W&B Organization dashboard at [INSERT URL].
2. Click the **Add user** button.
3. Add the user's email in the Email field.
4. Select the user role type you want to apply to the user. By default, all users are assigned a Member role.
5. Click the **Add new user** button.

[INSERT IMAGE]

:::info
Note that an option may be greyed out if there are no more seats in the license.
:::

An invite link will be sent to the user by email. The new admin or member will now have access to the W&B instance.

W&B uses third-party email server to send these invite emails. If your organization firewall rules prohibit from sending traffic outside the corporate network, W&B provides an option to set up internal SMTP server. Please refer to [these instructions](./smtp.md) to setup the SMTP server.

### Remove a user
1. Navigate to the W&B Organization dashboard at [INSERT URL].
2. Search for the user you want to modify in the search bar.
3. Click on the meatball menu (three horizontal dots).
4. Select **Remove user**.


### Change user roles

1. Navigate to the W&B Organization dashboard at [INSERT URL].
2. Search for the user you want to modify in the search bar.
3. Hover your mouse to the **Role** column. Click on the pencil icon that appears.
4. From the dropdown, select the new role you want to assign.




## Manage a team
Use a team home page as a central hub to explore projects, reports, and runs. Within the team home page there is a **Settings** tab. Use the Settings tab to manage members, set a team avatar, adjust privacy settings, set up alerts, track usage, and more. For more information, see the [Team settings](../app/settings-page/team-settings.md) page.

:::tip
Admins can add and remove team members. A team member is invited by email by the team admin. A team member cannot invite other members.

For more information on team roles and permissions, [see Team Roles and Permissions](../app/features/teams.md#team-roles-and-permissions).
:::

### Create a team

1. Navigate to the W&B Organization dashboard at [INSERT URL].
2. Select the **Create new team** button on the left navigation panel.
3. A modal will appear. Prove a name for your team in the **Team name** field. 
4. Select a storage type. 
5. Click on the **Create team** button.

This will redirect you to a newly created Team home page. 

### Team roles
When you invite a user to a team you can assign them one of the following roles:

| Role      | Definition                                                                                                                                                                                                                                                                                       |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Admin     | A team member who can add and remove other admins and members of the team.                                                                                                                                                                                                                       |
| Member    | A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.                                                                                                                                                                        |
| View-Only | A view-only member of your team, invited by email by the team admin. A view-only member only has read access to the team and its contents.                                                                                                                                                       |
| Service   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure that the environment variable `WANDB_USERNAME` is set to correctly attribute runs to the appropriate user. |


### Invite members to a team
Use the Team's settings page to invite members.

:::info
Members must first be part of the instance before they can be invited to a team.
:::

1. Navigate to the Team's Settings page.
2. Select the **Members** tab.
3. Enter an email or W&B username in the search bar.
4. Once you have found the user, click the **Invite** button.


### Remove members from a team

Use the Team's settings page to remove members.

1. Navigate to the Team's settings page.
2. Select the Delete button next the to member's name.

:::info
W&B runs logged by team members remain after a team member is removed.
:::


## View organization usage of W&B
Use the organization dashboard to get a holistic view of members that belong to your organization, how members of your organization use W&B, along with properties such as:

* Name: The name of the user and their W&B username.
* Last active: The time, in hours, the user last used W&B. Or status of user if pending an invitation.
* Role: The role of the user. [LINK to team role section]
* Email: The email of the user.
* Team: The names of teams the user belongs to.


View your organization board at [INSERT link and/or pattern]

### View the status of a user
The **Last Active** column shows if a user is pending an invitation or an active user.  A user is one of three states:

* Pending invitation:
* Active: 
* Deactivated: 

The **Role** column will display **Deactivated** if a user was deactivated. 

[INSERT IMAGE]


### View user activity
The **Last Active** column describes the time, in hours, the user last used W&B. A user is considered active if that user logs in to W&B, uses the SDK to track an experiment, views information on the W&B app, or interacts with the server in any way.

Hover your mouse over the Last Active entry for a user. A tooltip will appear and describe a summary of when that user was added along with the last time that user was active.

[INSERT IMAGE]

### View active users over time
Use the **Users active over time**  plot in the Organization dashboard to get an aggregate overview of how many users are active over time. 

[INSERT image]

You can use the dropdown menu to filer results based on days, months, or all time.

