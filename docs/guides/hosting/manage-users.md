# User Management
Weights & Biases strongly recommends and encourages user management with Single Sign-On (SSO). To learn more about how to setup SSO with W&B Server, refer to the [SSO Configuration documentation](./sso.md).



:::tip
Users are classified as either an *admin* or *member* when using W&B. Admins can add and remove other admins or team members. A team member is invited by email by the team admin. A team member cannot invite other members. 

For more information on roles and permissions, [see Team Roles and Permissions](../app/features/teams#team-roles-and-permissions). 
:::

## Instance Admins
The first user to sign up to W&B, after you have deployed W&B Server, will automatically be assigned admin permissions. The admin can then add additional users to the instance and create teams.

## Invite Users

Invite fellow admin or members from the `https://<YOUR-WANDB-URL>/admin/users` page.

1.  Navigate to `https://<YOUR-WANDB-URL>/admin/users`.

<!-- ![Screen Shot 2023-01-09 at 10.12.48 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9be8ae4-be4b-479a-b8b1-48c43e941f29/Screen_Shot_2023-01-09_at_10.12.48_PM.png) -->

![](/images/hosting/invite_users.png)

2. Click on **Add User**.

<!-- ![Screen Shot 2023-01-09 at 10.13.29 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf4dd778-5185-4503-8653-94388eae2e5b/Screen_Shot_2023-01-09_at_10.13.29_PM.png) -->

3. Enter the user's email. By default, all users are invited as Members. If you need to invite someone as an instance Admin, toggle the **Admin option** and click **Submit**.

<!-- ![Screen Shot 2023-01-09 at 10.16.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1428275-5ae0-4a36-8c1b-99248d7a7584/Screen_Shot_2023-01-09_at_10.16.04_PM.png) -->

An invite link will be sent to the user by email. The new admin or member will now have access to the W&B instance. 

Set up internal SMTP server if your organization has firewall rules that prevent sending email invites. 

<!-- To do: Add this doc -->
<!-- Refer to SMTP configuration documentation for instructions on how to do this. -->

## Create Teams
Navigate to `https://<YOUR-WANDB-URL>/admin/users` to create a new W&B Team.

1. Go to `https://<YOUR-WANDB-URL>/admin/users` page and click on **Teams**.

<!-- ![Screen Shot 2023-01-09 at 10.22.50 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d59520c-4a00-4596-9e2e-428b1b53c589/Screen_Shot_2023-01-09_at_10.22.50_PM.png) -->

2. Click on **New Team** and enter a name for the team in the **Team name** field. 

<!-- ![Screen Shot 2023-01-09 at 10.25.10 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/180f26ae-fa96-4dc4-b421-f9676ff73477/Screen_Shot_2023-01-09_at_10.25.10_PM.png) -->


Each team has its own profile page. Navigate to `https://<YOUR-WANDB-URL>/<team-name>` to view a team's profile page.


<!-- ![Screen Shot 2023-01-09 at 10.29.14 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7dbd7cac-9300-4a48-a67c-a696548b0153/Screen_Shot_2023-01-09_at_10.29.14_PM.png) -->

## Managing Team Settings

The Team home page includes the option for Team settings, which allows you to manage members, set a team avatar, adjust privacy settings, set up alerts, track usage, and more. For more information, see the [Team settings](../app/settings-page/team-settings.md) page.

## Invite members to a team

:::info
Members must first be part of the instance before they can be invited to a team. For instructions on inviting users to the instance, refer to the [Invite Users](#invite-users) section.
:::

When you invite a user to a team you can assign them one of the following roles:

| Role | Definition |
| ----- | ----- |
| Admin | A team member who can add and remove other admins and members of the team.|
| Member| A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.|
|Service | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure that the environment variable `WANDB_USERNAME` is set to correctly attribute runs to the appropriate user.|

<!-- **Admin**: A team member who can add and remove other admins and members of the team.

**Member**: A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.

**Service**: A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure that the environment variable `WANDB_USERNAME` is set to correctly attribute runs to the appropriate user. -->

<!-- ![Screen Shot 2023-01-09 at 10.48.49 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2eb67576-e0c5-4951-95ba-7a6fa49a8d68/Screen_Shot_2023-01-09_at_10.48.49_PM.png) -->

## Remove members from a team
Use the Team's settings page to remove members.

1. Navigate to the team settings page.
2. Select the Delete button next the to member's name.

:::info
W&B runs logged by team members remain after a team member is removed.
:::