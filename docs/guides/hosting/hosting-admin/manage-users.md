# User Management in W&B Server

Weights & Biases strongly recommends and encourages user management via Single Sign-On (SSO). To know more about how to setup SSO with W&B Server, please refer to the SSO Configuration documentation.

### Instance Admins

Upon successful deployment of the W&B server, the first user to sign up will automatically be assigned admin permissions. This individual will have the ability to add additional users to the instance and create teams.

### Inviting Users

In W&B, users can be classified as either admin or member. Instance admins have the ability to invite users with either role from the `https://<YOUR-WANDB-URL>/admin/users` page.

Step 1: Navigate to https://<YOUR-WANDB-URL>/admin/users

![Screen Shot 2023-01-09 at 10.12.48 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9be8ae4-be4b-479a-b8b1-48c43e941f29/Screen_Shot_2023-01-09_at_10.12.48_PM.png)

Step 2: Click on Add User

![Screen Shot 2023-01-09 at 10.13.29 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cf4dd778-5185-4503-8653-94388eae2e5b/Screen_Shot_2023-01-09_at_10.13.29_PM.png)

Step 3: Enter the user's email. By default, all users are invited as Members. If you need to invite someone as an instance Admin, toggle the Admin option and click Submit.

![Screen Shot 2023-01-09 at 10.16.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1428275-5ae0-4a36-8c1b-99248d7a7584/Screen_Shot_2023-01-09_at_10.16.04_PM.png)

An invite link will be sent to the user, allowing them access to the W&B instance. If your organization has firewall rules that prevent the sending of email invites, you can set up an internal SMTP server. Refer to SMTP configuration documentation for instructions on how to do this.

### Creating Teams

In order to create new teams in W&B, simply navigate to the `https://<YOUR-WANDB-URL>/admin/users` page and click on Teams.

![Screen Shot 2023-01-09 at 10.22.50 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d59520c-4a00-4596-9e2e-428b1b53c589/Screen_Shot_2023-01-09_at_10.22.50_PM.png)

Click on `New Team` and enter the Team Name

![Screen Shot 2023-01-09 at 10.25.10 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/180f26ae-fa96-4dc4-b421-f9676ff73477/Screen_Shot_2023-01-09_at_10.25.10_PM.png)

Each team has its own profile page, which can be accessed by clicking on the team name from the Teams page or by navigating to `https://<YOUR-WANDB-URL>/<team-name>`.

![Screen Shot 2023-01-09 at 10.29.14 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7dbd7cac-9300-4a48-a67c-a696548b0153/Screen_Shot_2023-01-09_at_10.29.14_PM.png)

### Managing Team Settings

The Team home page includes the option for Team settings, which allows you to manage members, set a team avatar, adjust privacy settings, set up alerts, track usage, and more.

**Inviting members to a team**

> Kindly note that users must first be part of the instance before they can be invited to a team. For instructions on inviting users to the instance, refer to the [Inviting Users](https://www.notion.so/Docs-505-User-management-in-W-B-a691dddcd2f74195a8b89aece7582390) section.

When inviting users to a team, you have the following options to assign to them:

**Admin**: A team member who can add and remove other admins and members of the team.

**Member**: A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.

**Service**: A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure that the environment variable `WANDB_USERNAME` is set to correctly attribute runs to the appropriate user.

![Screen Shot 2023-01-09 at 10.48.49 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2eb67576-e0c5-4951-95ba-7a6fa49a8d68/Screen_Shot_2023-01-09_at_10.48.49_PM.png)

**Removing members from a team**

When a team member leaves, team admins can open the team settings page and click the delete button next to the departing member's name. Any runs that they logged to the team will remain even after a user has been removed.
