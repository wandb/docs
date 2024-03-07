---
description: Restricted Projects for collaborating on AI workflows with sensitive data
displayed_sidebar: default
---

# Restricted projects

## Overview

AI practitioners manage their workflows within W&B projects that or organized in a W&B team. There are different project scopes possible in the order of most public to most private - _Open_, _Public_, _Team_ and _Restricted_. Restricted Projects are a type of project scope that's meant for practitioners who want to collaborate on workflows related to sensitive or confidential data.

When you create a restricted project within a team, you can invite or add specific members from the team to collaborate on relevant experiments, artifacts, reports etc. Unlike other project scopes, all members of a team do not get implicit access to a restricted project. At the same time, team admins can join restricted projects to monitor all team activity. Each member inherits their team scoped role when you add them to your restricted project.

![](/images/hosting/restricted_project_1.png)

![](/images/hosting/restricted_project_2.png)

:::info
When a team admin enables the team privacy setting `Make all future team projects private (public sharing not allowed)`, that disables the _Open_ and _Public_ project scopes in that team. One can only use _Team_ and _Restricted_ scopes within such a team.
:::

:::info
If you want to use a team-level service account in a restricted project, you should invite or add that specifically to the project. Otherwise a team-level service account can not access a restricted project by default.
:::

## Editing a restricted project

If you are the owner of a restricted project you or a team admin can edit it by using the `Edit project details` button in the project overview tab, and add or remove team members from the opened interface. You can also change the scope of the project to _Team_ if needed.

![](/images/hosting/restricted_project_edit.png)

:::caution
If you change the scope of a restricted project to _Team_, it's accessible to all members of the team. Similarly, if you change the scope of a team project to _Restricted_, all members of the team lose access to that project unless you add them specifically. So make the scope change after deliberating with your admins or security team.
:::



