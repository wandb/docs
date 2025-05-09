---
menu:
  default:
    identifier: privacy-settings
    parent: w-b-platform
title: Configure privacy settings
weight: 4
---

Organization and Team admins can configure a set of privacy settings at the organization and team scopes respectively. When configured at the organization scope, organization admins enforce those settings for all teams in that organization.

{{% alert %}}
W&B recommends organization admins to enforce a privacy setting only after communicating that in advance to all team admins and users in their organization. This is to avoid unexpected changes in their workflows.
{{% /alert %}}

## Configure privacy settings for a team

Team admins can configure privacy settings for their respective teams from within the `Privacy` section of the team **Settings** tab. Each setting is configurable as long as it's not enforced at the organization scope:

* Hide this team from all non-members
* Make all future team projects private (public sharing not allowed)
* Allow any team member to invite other members (not just admins)
* Turn off public sharing to outside of team for reports in private projects. This turns off existing magic links.
* Allow users with matching organization email domain to join this team.
    * This setting is applicable only to [SaaS Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}). It's not available in [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) instances.
* Enable code saving by default.

## Enforce privacy settings for all teams

Organization admins can enforce privacy settings for all teams in their organization from within the `Privacy` section of the **Settings** tab in the account or organization dashboard. If organization admins enforce a setting, team admins are not allowed to configure that within their respective teams.

* Enforce team visibility restrictions
    * Enable this option to hide all teams from non-members
* Enforce privacy for future projects
    * Enable this option to enforce all future projects in all teams to be private or [restricted]({{< relref "./iam/access-management/restricted-projects.md" >}})
* Enforce invitation control
    * Enable this option to prevent non-admins from inviting members to any team
* Enforce report sharing control
    * Enable this option to turn off public sharing of reports in private projects and deactivate existing magic links
* Enforce team self joining restrictions
    * Enable this option to restrict users with matching organization email domain from self-joining any team
    * This setting is applicable only to [SaaS Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}). It's not available in [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) or [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) instances.
* Enforce default code saving restrictions
    * Enable this option to turn off code saving by default for all teams