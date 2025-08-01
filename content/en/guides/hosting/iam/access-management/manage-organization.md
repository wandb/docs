---
menu:
  default:
    identifier: manage-organization
    parent: access-management
title: Manage your organization
weight: 1
---
As an admin of an organization you can [manage individual users]({{< relref "#add-and-manage-users" >}}) within your organization and [manage teams]({{< relref "#add-and-manage-teams" >}}). 

As a team admin you can [manage teams]({{< relref "#add-and-manage-teams" >}}).

{{% alert %}}
The following workflow applies to users with instance admin roles. Reach out to an admin in your organization if you believe you should have instance admin permissions. 
{{% /alert %}}

If you are looking to simplify user management in your organization, refer to [Automate user and team management]({{< relref "../automate_iam.md" >}}).

<!-- W&B assigns an Admin role to new users within an organization by default.  -->

## Change the name of your organization
{{% alert %}}
The following workflow only applies to W&B Multi-tenant SaaS Cloud.
{{% /alert %}}

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
3. Within the **Settings** tab, select **General**.
4. Select the **Change name** button.
5. Within the modal that appears, provide a new name for your organization and select the **Save name** button.

## Add and manage users

As an admin, use your organization's dashboard to:
- Invite or remove users.
- Assign or update a user's organization role, and create custom roles.
- Assign the billing admin.

There are several ways an organization admin can add users to an organization:

1. Member-by-invite
2. Auto provisioning with SSO
3. Domain capture

### Seats and pricing

The proceeding table summarizes how seats work for Models and Weave:

| Product |Seats | Cost based on |
| ----- | ----- | ----- |
| Models | Pay per set | How many Models paid seats you have, and how much usage you’ve accrued determines your overall subscription cost. Each user can be assigned one of the three available seat types: Full, Viewer, and No-Access |
| Weave | Free  | Usage based |

### Invite a user

admins can invite users to their organization, as well as specific teams within the organization.

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Users**.
3. Select **Invite new user**.
4. In the modal that appears, provide the email or username of the user in the **Email or username** field.
5. (Recommended) Add the user to a team from the **Choose teams** dropdown menu.
6. From the **Select role** dropdown, select the role to assign to the user. You can change the user's role at a later time. See the table listed in [Assign a role]({{< relref "#assign-or-update-a-team-members-role" >}}) for more information about possible roles.
7. Choose the **Send invite** button.

W&B sends an invite link using a third-party email server to the user's email after you select the **Send invite** button. A user can access your organization once they accept the invite.
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
2. Select the **Add user** button
3. Within the modal that appears, provide the email of the new user in the **Email** field.
4. Select a role to assign to the user from the **Role** dropdown. You can change the user's role at a later time. See the table listed in [Assign a role]({{< relref "#assign-or-update-a-team-members-role" >}}) for more information about possible roles.
5. Check the **Send invite email to user** box if you want W&B to send an invite link using a third-party email server to the user's email.
6. Select the **Add new user** button.
{{% /tab %}}
{{< /tabpane >}}

### Auto provision users

A W&B user with matching email domain can sign in to your W&B Organization with Single Sign-On (SSO) if you configure SSO and your SSO provider permits it. SSO is available for all Enterprise licenses.

{{% alert title="Enable SSO for authentication" %}}
W&B strongly recommends and encourages that users authenticate using Single Sign-On (SSO). Reach out to your W&B team to enable SSO for your organization. 

To learn more about how to setup SSO with Dedicated cloud or Self-managed instances, refer to [SSO with OIDC]({{< relref "../authentication/sso.md" >}}) or [SSO with LDAP]({{< relref "../authentication/ldap.md" >}}).{{% /alert %}}


W&B assigned auto-provisioning users "Member" roles by default. You can change the role of auto-provisioned users at any time.

Auto-provisioning users with SSO is on by default for Dedicated cloud instances and Self-managed deployments. You can turn off auto provisioning. Turning auto provisioning off enables you to selectively add specific users to your W&B organization.

The proceeding tabs describe how to turn off SSO based on deployment type:

{{< tabpane text=true >}}
{{% tab header="Dedicated cloud" value="dedicated" %}}
Reach out to your W&B team if you are on Dedicated cloud instance and you want to turn off auto provisioning with SSO.
{{% /tab %}}

{{% tab header="Self-managed" value="self_managed" %}}
Use the W&B Console to turn off auto provisioning with SSO:

1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
2. Choose **Security** 
3. Select the **Disable SSO Provisioning** to turn off auto provisioning with SSO.

<!-- For Self-managed deployments, you can configure the setting `DISABLE_SSO_PROVISIONING=true` to turn off auto provisioning with SSO.  -->

{{% /tab %}}
{{< /tabpane >}}

{{% alert title="" %}}
Auto provisioning with SSO is useful for adding users to an organization at scale because organization admins do not need to generate individual user invitations.
{{% /alert %}}

### Create custom roles
{{% alert %}}
An Enterprise license is required to create or assign custom roles on Dedicated cloud or Self-managed deployments.
{{% /alert %}}

Organization admins can compose a new role based on either the View-Only or Member role and add additional permissions to achieve fine-grained access control. Team admins can assign a custom role to a team member. Custom roles are created at the organization level but are assigned at the team level.

To create a custom role:

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
1. Click **Roles**.
1. In the **Custom roles** section, click **Create a role**.
1. Provide a name for the role. Optionally provide a description.
1. Choose the role to base the custom role on, either **Viewer** or **Member**.
1. To add permissions, click the **Search permissions** field, then select one or more permissions to add.
1. Review the **Custom role permissions** section, which summarizes the permissions the role has.
1. Click **Create Role**.
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
Use the W&B Console to turn off auto provisioning with SSO:

1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
1. In the **Custom roles** section, click **Create a role**.
1. Provide a name for the role. Optionally provide a description.
1. Choose the role to base the custom role on, either **Viewer** or **Member**.
1. To add permissions, click the **Search permissions** field, then select one or more permissions to add.
1. Review the **Custom role permissions** section, which summarizes the permissions the role has.
1. Click **Create Role**.

{{% /tab %}}
{{< /tabpane >}}

A team admin can now assign the custom role to members of a team from the [Team settings]({{< relref "#invite-users-to-a-team" >}}).

### Domain capture
Domain capture helps your employees join the your companies organization to ensure new users do not create assets outside of your company jurisdiction. 

{{% alert title="Domains must be unique" %}}
Domains are unique identifiers. This means that you can not use a domain that is already in use by another organization. 
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
Domain capture lets you automatically add people with a company email address, such as  `@example.com`, to your W&B SaaS cloud organization. This helps all your employees join the right organization and ensures that new users do not create assets outside of your company jurisdiction. 

This table summarizes the behavior of new and existing users with and without domain capture enabled:

| | With domain capture | Without domain capture |
| ----- | ----- | ----- |
| New users | Users who sign up for W&B from verified domains are automatically added as members to your organization’s default team. They can choose additional teams to join at sign up, if you enable team joining. They can still join other organizations and teams with an invitation. | Users can create W&B accounts without knowing there is a centralized organization available. | 
| Invited users | Invited users automatically join your organization when accepting your invite. Invited users are not automatically added as members to your organization’s default team. They can still join other organizations and teams with an invitation. | Invited users automatically join your organization when accepting your invite. They can still join other organizations and teams with an invitation.| 
| Existing users | Existing users with verified email addresses from your domains can join your organization’s teams within the W&B App. All data that existing users create before joining your organization remains. W&B does not migrate the existing user's data. | Existing W&B users may be spread across multiple organizations and teams.|

To automatically assign non-invited new users to a default team when they join your organization:

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Choose the **Claim domain** button within **Domain capture**.
5. Select the team that you want new users to automatically join from the **Default team** dropdown. If no teams are available, you'll need to update team settings. See the instructions in [Add and manage teams]({{< relref "#add-and-manage-teams" >}}).
6. Click the **Claim email domain** button.

You must enable domain matching within a team's settings before you can automatically assign non-invited new users to that team.

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`. Where `<team-name>` is the name of the team you want to enable domain matching.
2. Select **Team settings** in the global navigation on the left side of the team's dashboard.
3. Within the **Privacy** section, toggle the "Recommend new users with matching email domains join this team upon signing up" option.

{{% /tab %}}
{{% tab header="Dedicated or Self-managed" value="dedicated" %}}
Reach out to your W&B Account Team if you use Dedicated or Self-managed deployment type to configure domain capture. Once configured, your W&B SaaS instance automatically prompts users who create a W&B account with your company email address to contact your admin to request access to your Dedicated or Self-managed instance.

| | With domain capture | Without domain capture |
| ----- | ----- | -----|
| New users | Users who sign up for W&B on SaaS cloud from verified domains are automatically prompted to contact an admin with an email address you customize. They can still create an organizations on SaaS cloud to trial the product. | Users can create W&B SaaS cloud accounts without learning their company has a centralized dedicated instance. | 
| Existing users | Existing W&B users may be spread across multiple organizations and teams.| Existing W&B users may be spread across multiple organizations and teams.|
{{% /tab %}}
{{< /tabpane >}}


### Assign or update a user's role

Every member in an Organization has an organization role and seat for both W&B Models and Weave. The type of seat they have determines both their billing status and the actions they can take in each product line.

You initially assign an organization role to a user when you invite them to your organization. You can change any user's role at a later time.

A user within an organization can have one of the proceeding roles:

| Role | Descriptions |
| ----- | ----- |
| admin| A instance admin who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends ensuring there is more than one admin in the event that your admin is unavailable. |
| Member | A regular user of the organization, invited by an instance admin. A organization member cannot invite other users or manage existing users in the organization. |
| Viewer (Enterprise-only feature) | A view-only user of your organization, invited by an instance admin. A viewer only has read access to the organization and the underlying teams that they are a member of. |
|Custom Roles (Enterprise-only feature) | Custom roles allow organization admins to compose new roles by inheriting from the preceding View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams.|

To change a user's role:

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
4. Select a role from the **TEAM ROLE** dropdown next to the name of the user.

### Assign or update a user's access

A user within an organization has one of the proceeding model seat or weave access types: full, viewer, or no access.  

| Seat type | Description |
| ----- | ----- |
| Full | Users with this role type have full permissions to write, read, and export data for Models or Weave. |
| Viewer | A view-only user of your organization. A viewer only has read access to the organization and the underlying teams that they are a part of, and view only access to Models or Weave. |
| No access | Users with this role have no access to the Models or Weave products. |

Model seat type and weave access type are defined at the organization level, and inherited by the team. If you want to change a user's seat type, navigate to the organization settings and follow the proceeding steps:

1. For SaaS users, navigate to your organization's settings at `https://wandb.ai/account-settings/<organization>/settings`. Ensure to replace the values enclosed in angle brackets (`<>`) with your organization name. For other Dedicated and Self-managed deployments, navigate to `https://<your-instance>.wandb.io/org/dashboard`.
2. Select the **Users** tab.
3. From the **Role** dropdown, select the seat type you want to assign to the user.

{{% alert %}}
The organization role and subscription type determines which seat types are available within your organization.
{{% /alert %}}

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
5. Under the **Billing admin** column, choose the user you want to assign as the billing admin.


## Add and manage teams
Use your organization's dashboard to create  and manage teams within your organization. An organization admin or a team admin can:
- Invite users to a team or remove users from a team.
- Manage a team member's roles.
- Automate the addition of users to a team when they join your organization.
- Manage team storage with the team's dashboard at `https://wandb.ai/<team-name>`.

### Create a team

Use your organization's dashboard to create a team:

1. Navigate to https://wandb.ai/home.
2. Select **Create a team to collaborate** on the left navigation panel underneath **Teams**.
{{< img src="/images/hosting/create_new_team.png" alt="Create new team" >}}
3. Provide a name for your team in the **Team name** field in the modal that appears. 
4. Choose a storage type. 
5. Select the **Create team** button.

After you select **Create team** button, W&B redirects you to a new team page at `https://wandb.ai/<team-name>`. Where `<team-name>` consists of the name you provide when you create a team.

Once you have a team, you can add users to that team.

### Invite users to a team

Invite users to a team in your organization. Use the team's dashboard to invite users using their email address or W&B username if they already have a W&B account.

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the global navigation on the left side of the dashboard.
{{< img src="/images/hosting/team_settings.png" alt="Team settings" >}}
3. Select the **Users** tab.
4. Choose on **Invite a new user**.
5. Within the modal that appears, provide the email of the user in the **Email or username** field and select the role to assign to that user from the **Select a team** role dropdown. For more information about roles a user can have in a team, see [Team roles]({{< relref "#assign-or-update-a-team-members-role" >}}).
6. Choose on the **Send invite** button.

By default, only a team or instance admin can invite members to a team. To change this behavior, refer to [Team settings]({{< relref "/guides/models/app/settings-page/team-settings.md#privacy" >}}).

In addition to inviting users manually with email invites, you can automatically add new users to a team if the new user's [email matches the domain of your organization]({{< relref "#domain-capture" >}}).

### Match members to a team organization during sign up

Allow new users within your organization discover Teams within your organization when they sign-up. New users must have a verified email domain that matches your organization's verified email domain. Verified new users can view a list of verified teams that belong to an organization when they sign up for a W&B account.

An organization admin must enable domain claiming. To enable domain capture, see the steps described in [Domain capture]({{< relref "#domain-capture" >}}).


### Assign or update a team member's role


1. Select the account type icon next to the name of the team member. 
2. From the drop-down, choose the account type you want that team member to posses.

This table lists the roles you can assign to a member of a team:

| Role   |   Definition   |
|-----------|---------------------------|
| admin    | A user who can add and remove other users in the team, change user roles, and configure team settings.   |
| Member    | A regular user of a team, invited by email or their organization-level username by the team admin. A member user cannot invite other users to the team.  |
| View-Only (Enterprise-only feature) | A view-only user of a team, invited by email or their organization-level username by the team admin. A view-only user only has read access to the team and its contents.  |
| Service (Enterprise-only feature)   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure to set the environment variable `WANDB_USERNAME`  to correctly attribute runs to the appropriate user. |
| Custom Roles (Enterprise-only feature)   | Custom roles allow organization admins to compose new roles by inheriting from the preceding View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams. Refer to the [custom roles announcement](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details. |

{{% alert %}}
Only enterprise licenses on Dedicated cloud or Self-managed deployment can assign custom roles to members in a team.
{{% /alert %}}

### Remove users from a team
Remove a user from a team using the team's dashboard. W&B preserves runs created in a team even if the member who created the runs is no longer on that team.

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the left navigation bar.
3. Select the **Users** tab.
4. Hover your mouse next to the name of the user you want to delete. Select the ellipses or three dots icon (**...**) when it appears. 
5. From the dropdown, select **Remove user**. 


<!-- To do as a follow up -->
<!-- ### Manage team storage

## Create and assign custom roles

## Privacy -->