---
description: Restricted Projects for collaborating on AI workflows with sensitive data
displayed_sidebar: default
---

# Restricted Projects

## Overview

AI practioners manage their workflows within W&B projects that or organized in a W&B team. There are different project scopes possible in the order of most public to most private - _Open_, _Public_, _Team_ and _Restricted_. Restricted Projects are a type of project scope that's meant for practioners who want to collaborate on workflows related to sensitive or confidential data.

When you create a restricted project within a team, you can invite or add specific members from the team to collaborate on relevant experiments, artifacts, reports etc. Unlike other project scopes, all members of a team do not get implicit access to a restricted project. At the same time, team admins are allowed to join restricted projects in order to ensure that all activity within a team can be monitored and tracked. Each member inherits their team scoped role when you add them to your restricted project.

![](/images/hosting/restricted_project_1.png)

![](/images/hosting/restricted_project_2.png)

:::info
When the team privacy setting `Make all future team projects private (public sharing not allowed)` is enabled, then _Open_ and _Public_ project scopes are disabled, and only _Team_ and _Restricted_ scopes are allowed within that team.
:::

:::info
If you want to use a team-level service account in a restricted project, you will have to invite or add that as well specifically to that project. Otherwise a team-level service account will not be able to access the restricted project.
:::

## Editing a restricted project

If you are the owner of a restricted project, you or a team admin can edit it by using the `Edit project details` button in the project overview tab, and add or remove team members from the opened interface. You can also change the scope of the project to _Team_ if needed.

![](/images/hosting/restricted_project_edit.png)

:::caution
If you change the scope of a restricted project to _Team_, it's accessible to all members of the team. Similarly, if you change the scope of a team project to _Restricted_, all members of the team lose access to that project unless they are added specifically. So make the scope change after deliberating with your admins or security team.
:::



