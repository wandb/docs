---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Manage your organization

As an administrator of an organization you can [manage individual users](#add-and-manage-users) within your organization and [manage teams](#add-and-manage-teams). 

As a team administrator you can [manage teams](#add-and-manage-teams).

:::info
The following workflow applies to users with instance administrator (admin) roles. Reach out to an administrator in your organization if you believe you should have instance administrator permissions. 
:::

If you are looking to simplify user management in your organization, refer to [Automate user and team management](./automate_iam.md).

<!-- W&B assigns an Admin role to new users within an organization by default.  -->

## Change the name of your organization
:::info
The following workflow only applies to W&B Multi-tenant SaaS Cloud.
:::

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
3. Within the **Settings** tab, select **General**.
4. Select the **Change name** button.
5. Within the modal that appears, provide a new name for your organization and select the **Save name** button.

## Add and manage users

As an administrator, use your organization's dashboard to invite users, assign or update a user's role, remove users from your organization, assign the billing administrator, and more.

There are several ways an organization administrator can add users to an organization:

1. Member-by-invite
2. Auto provisioning with SSO
3. Domain capture

### Invite a user

Administrators can invite users to their organization, as well as specific teams within the organization.

<Tabs
  defaultValue="saas"
  values={[
    {label: 'Multi-tenant SaaS Cloud', value: 'saas'},
    {label: 'Dedicated or Self Managed', value: 'dedicated'},
  ]}>
  <TabItem value="saas">

1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Users**.
3. Select **Invite new user**.
4. In the modal that appears, provide the email or username of the user in the **Email or username** field.
5. (Recommended) Add the user to a team from the **Choose teams** dropdown menu.
6. From the **Select role** dropdown, select the role to assign to the user. You can change the user's role at a later time. See the table listed in [Assign a role](#assign-or-update-a-team-members-role) for more information about possible roles.
7. Choose the **Send invite** button.

W&B sends an invite link using a third-party email server to the user's email after you select the **Send invite** button. A user can access your organization once they accept the invite.

  </TabItem>
  <TabItem value="dedicated">

1. Navigate to `https://wandb.io/org/dashboard`
2. Select the **Add user** button
3. Within the modal that appears, provide the email of the new user in the **Email** field.
4. Select a role to assign to the user from the **Role** dropdown. You can change the user's role at a later time. See the table listed in [Assign a role](#assign-or-update-a-team-members-role) for more information about possible roles.
5. Check the **Send invite email to user** box if you want W&B to send an invite link using a third-party email server to the user's email.
6. Select the **Add new user** button.

  </TabItem>
</Tabs>



:::tip enable SSO for authentication
W&B strongly recommends and encourages that users authenticate using Single Sign-On (SSO). Reach out to your W&B team to enable SSO for your organization. 

To learn more about how to setup SSO with Dedicated cloud or Self-managed instances, refer to [SSO with OIDC](./sso.md) or [SSO with LDAP](./ldap.md).
:::

### Auto provision users
:::info
The following workflow only applies to Dedicated cloud instances and Self-Managed deployments.
:::

A person in your company (someone who has the same domain as your company) can sign in to your W&B Organization with Single Sign-On (SSO) if SSO is set up and the SSO provider permits it.


W&B assigned auto-provisioning users "Member" roles by default. You can change the role of auto-provisioned users at any time.

Auto-provisioning users with SSO is on by default for Dedicated cloud instances and Self-Managed deployments. You can turn off auto provisioning. Turning auto provisioning off enables you to selectively add specific users to your W&B organization.

The proceeding tabs describe how to turn off SSO based on deployment type:

<Tabs
  defaultValue="dedicated"
  values={[
    {label: 'Dedicated Cloud', value: 'dedicated'},
    {label: 'Self Manged', value: 'self_managed'},
  ]}>
  <TabItem value="dedicated">

Reach out to your W&B team if you are on Dedicated Cloud instance and you want to turn off auto provisioning with SSO.

  </TabItem>
  <TabItem value="self_managed">

Use the W&B Console to turn off auto provisioning with SSO:

1. Navigate to `https://wandb.io/console/settings/`
2. Choose **Security** 
3. Select the **Disable SSO Provisioning** to turn off auto provisioning with SSO.

<!-- For Self-Managed deployments, you can configure the setting `DISABLE_SSO_PROVISIONING=true` to turn off auto provisioning with SSO.  -->


  </TabItem>
</Tabs>



:::tip 
Auto provisioning with SSO is useful for adding users to an organization at scale because organization administrators do not need to generate individual user invitations.
:::


### Domain capture
Domain capture helps your employees join the your companies organization to ensure new users do not create assets outside of your company jurisdiction. 

:::note Domains must be unique
Domains are unique identifiers. This means that you can not use a domain that is already in use by another organization. 
:::

<Tabs
  defaultValue="saas"
  values={[
    {label: 'Multi-tenant SaaS Cloud', value: 'saas'},
    {label: 'Dedicated or Self Managed', value: 'dedicated'},
  ]}>
  <TabItem value="saas">

Domain capture lets you automatically add people with a company email address, such as  `@example.com`, to your W&B SaaS cloud organization. This helps all your employees join the right organization and ensures that new users do not create assets outside of your company jurisdiction. 


The proceeding table summarizes the behavior of new and existing users with and without domain capture enabled:

| | With domain capture | Without domain capture |
| ----- | ----- | ----- |
| New users | Users who sign up for W&B from verified domains are automatically added as members to your organization’s default team. They can choose additional teams to join at sign up, if team joining is enabled for these teams. They can still join other organizations and teams with an invitation. | Users can create W&B accounts without knowing there is a centralized organization available. | 
| Invited users | Invited users automatically join your organization when accepting your invite. Invited users are not automatically added as members to your organization’s default team. They can still join other organizations and teams with an invitation. | Invited users automatically join your organization when accepting your invite. They can still join other organizations and teams with an invitation.| 
| Existing users | Existing users with verified email addresses from your domains can join your organization’s teams within the W&B App. All data that existing users create before joining your organization will remain. No data will be migrated. | Existing W&B users may be spread across multiple organizations and teams.|

To automatically assign non-invited new users to a default team when they join your organization:

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Choose the **Claim domain** button within **Domain capture**.
5. Select the team that you want new users to automatically join from the **Default team** dropdown. If no teams are available, you'll need to update team settings. See the instructions in [Add and manage teams](#add-and-manage-teams).
6. Click the **Claim email domain** button.

You must enable domain matching within a team's settings before you can automatically assign non-invited new users to that team.

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`. Where `<team-name>` is the name of the team you want to enable domain matching.
2. Select **Team settings** in the global navigation on the left side of the team's dashboard.
3. Within the **Privacy** section, toggle the "Recommend new users with matching email domains join this team upon signing up" option.


 </TabItem>
 
<TabItem value="dedicated">

Reach out to your W&B Account Team if you use Dedicated or Self-Managed deployment type to configure domain capture. Once configured, your W&B SaaS instance automatically prompts users who create a W&B account with your company email address to contact your administrator to request access to your Dedicated or Self-Managed instance.

| | With domain capture | Without domain capture |
| ----- | ----- |
| New users | Users who sign up for W&B on SaaS cloud from verified domains are automatically prompted to contact an admin via an email address you customize. They can still create an organizations on SaaS cloud to trial the product. | Users can create W&B SaaS cloud accounts without learning their company has a centralized dedicated instance. | 
| Existing users | Existing W&B users may be spread across multiple organizations and teams.| Existing W&B users may be spread across multiple organizations and teams.|

</TabItem>
</Tabs>


### Assign or update a user's role

You initially assign a role to a user when you invite them to your organization. You can change any user's role at a later time.

A user within an organization can have one of the proceeding roles:

| Role | Descriptions |
| ----- | ----- |
| administrator| A instance administrator who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends more than one administrator for an enterprise Dedicated cloud or Self-managed instances. |
| Member | A regular user of the organization, invited by an instance administrator. A organization user cannot invite other users or manage existing users in the organization. |
| Viewer | A view-only user of your organization, invited by an instance administrator. A viewer only has read access to the organization and the underlying teams that they are a part of.  |

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

### Assign the billing administrator
1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
5. Under the **Billing admin** column, choose the user you want to assign as the billing administrator.


## Add and manage teams
Use your organization's dashboard to create teams within your organization. Once an organization administrator creates a team, either the org administrator or team administrator can invite users to that team, assign or update a team member's role, automatically add new users to a team when they join your organization, remove users from a team, and manage team storage with the team's dashboard at `https://wandb.ai/<team-name>`.

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

In addition to inviting users manually with email invites, you can automatically add new users to a team if the new user's [email matches the domain of your organization](#domain-capture).

### Assign or update a team member's role
The proceeding table lists the roles you can assign to a member of a team:

| Role   |   Definition   |
|-----------|---------------------------|
| administrator    | A user who can add and remove other users in the team, change user roles, and configure team settings.   |
| Member    | A regular user of a team, invited by email or their organization-level username by the team administrator. A member user cannot invite other users to the team.  |
| View-Only (Enterprise-only feature) | A view-only user of a team, invited by email or their organization-level username by the team administrator. A view-only user only has read access to the team and its contents.  |
| Service (Enterprise-only feature)   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure to set the environment variable `WANDB_USERNAME`  to correctly attribute runs to the appropriate user. |
| Custom Roles (Enterprise-only feature)   | Custom roles allow organization administrators to compose new roles by inheriting from the preceding View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team administrators can then assign any of those custom roles to users in their respective teams. Refer to [this article](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details. |

<!-- :::note
W&B recommends to have more than one admin in a team. It is a best practice to ensure that admin operations can continue when the primary admin is not available.

Refer to [Team Service Account Behavior](../../app/features/teams.md#team-service-account-behavior) for more information.
::: -->


:::note
Only enterprise licenses on Dedicated Cloud or Self-managed deployment can assign custom roles to members in a team.
:::

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
