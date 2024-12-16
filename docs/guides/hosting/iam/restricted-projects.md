---
description: Manage project access using visibility scopes and project-level roles
title: Manage access control for projects
---

# Project visibility

Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it. 
You can use a combination of a couple of controls to configure the access level for any project within a W&B team. **Visibility scope** is the higher-level mechanism. Use that to control which groups of users can view or submit runs in a project. For a project with _Team_ or _Restricted_ visibility scope, you can then use **Project level roles** to control the level of access that each user has within the project.

{{% alert %}}
The owner of a project, a team admin, or an organization admin can set or edit a project's visibility.
{{% /alert %}}

## Visibility scopes

You can choose from the following project visibility scopes, ordered from most public to most private:

| Scope | Description | 
| ----- | ----- |
| Open | Anyone who knows about the project can view it and submit runs or reports.|
| Public | Anyone who knows about the project can view it. Only your team can submit runs or reports.|
| Team | Only members of the parent team can view the project and submit runs or reports. Anyone outside the team can not access the project.|
| Restricted | Only invited members from the parent team can view the project and submit runs or reports.|

{{% alert %}}
Set a project's scope to **Restricted** if you would like to collaborate on workflows related to sensitive or confidential data. When you create a restricted project within a team, you can invite or add specific members from the team to collaborate on relevant experiments, artifacts, reports, and so forth. 

Unlike other project scopes, all members of a team do not get implicit access to a restricted project. At the same time, team admins can join restricted projects if needed.
{{% /alert %}}

### Set visibility scope on a new or existing project

Set a project's visibility scope when you create a project or when editing it later.

{{% alert %}}
* Only the owner of the project or a team admin can set or edit its visibility scope.
* When a team admin enables **Make all future team projects private (public sharing not allowed)** within a team's privacy setting, that turns off **Open** and **Public** project visibility scopes for that team. In this case, your team can only use **Team** and **Restricted** scopes.
{{% /alert %}}

#### Set visibility scope when you create a new project

1. Navigate to your W&B organization on SaaS Cloud, Dedicated Cloud, or Self-managed instance.
2. Click the **Create a new project** button in the left hand sidebar's **My projects** section. Alternatively, navigate to the **Projects** tab of your team and click the **Create new project** button in the upper right hand corner.
3. After selecting the parent team and entering the name of the project, select the desired scope from the **Project Visibility** dropdown.
{{< img src="/images/hosting/restricted_project_add_new.gif" alt="" >}}

Complete the following step if you select **Restricted** visibility. 

4. Provide names of one or more W&B team members in the **Invite team members** field. Add only those members who are essential to collaborate on the project.
{{< img src="/images/hosting/restricted_project_2.png" alt="" >}}

{{% alert %}}
You can add or remove members in a restricted project later, from its **Users** tab.
{{% /alert %}}

#### Edit visibility scope of an existing project

1. Navigate to your W&B Project.
2. Select the **Overview** tab on the left column.
3. Click the **Edit Project Details** button on the upper right corner.  
4. From the **Project Visibility** dropdown, select the desired scope.
{{< img src="/images/hosting/restricted_project_edit.gif" alt="" >}}

Complete the following step if you select **Restricted** visibility. 

5. Go to the **Users** tab in the project, and click **Add user** button to invite specific users to the restricted project.

{{% alert color="secondary" %}}
* All members of a team lose access to a project if you change its visibility scope from **Team** to **Restricted**, unless you invite the required team members to the project.
* All members of a team get access to a project if you change its visibility scope from **Restricted** to **Team**.
* If you remove a team member from the user list for a restricted project, they lose access to that project.
{{% /alert %}}

### Other key things to note for restricted scope

* If you want to use a team-level service account in a restricted project, you should invite or add that specifically to the project. Otherwise a team-level service account can not access a restricted project by default.
* You can not move runs from a restricted project, but you can move runs from a non-restricted project to a restricted one.
* You can convert the visibility of a restricted project to only **Team** scope, irrespective of the team privacy setting **Make all future team projects private (public sharing not allowed)**.
* If the owner of a restricted project is not part of the parent team anymore, the team admin should change the owner to ensure seamless operations in the project.

## Project level roles

For the _Team_ or _Restricted_ scoped projects in your team, you can assign a specific role to a user, which could be different from that user's team level role. For example, if a user has _Member_ role at the team level, you can assign the _View-Only_, or _Admin_, or any available custom role to that user within a _Team_ or _Restricted_ scope project in that team.

{{% alert %}}
Project level roles are in preview on SaaS Cloud, Dedicated Cloud, and Self-managed instances.
{{% /alert %}}

### Assign project level role to a user

1. Navigate to your W&B Project.
2. Select the **Overview** tab on the left column.
3. Go to the **Users** tab in the project.
4. Click the currently assigned role for the pertinent user in the **Project Role** field, which should open up a dropdown listing the other available roles.
5. Select another role from the dropdown. It should save instantly.

{{% alert %}}
When you change the project level role for a user to be different from their team level role, the project level role includes a **\*** to indicate the difference.
{{% /alert %}}

### Other key things to note for project level roles

* By default, project level roles for all users in a _team_ or _restricted_ scoped project **inherit** their respective team level roles.
* You **can not** change the project level role of a user who has _View-only_ role at the team level.
* If the project level role for a user within a particular project **is same as** the team level role, and at some point if a team admin changes the team level role, the relevant project role is automatically changed to track the team level role.
* If you change the project level role for a user within a particular project such that **it is different from** the team level role, and at some point if a team admin changes the team level role, the relevant project level role remains as is.
* If you re-add a user to a _restricted_ project some time after their removal, and if their project-level role was different from their team-level role before removal, they inherit the team-level role after the re-addition, due to the default role inheritance behavior. If necessary, you must reassign the project-level role to reinstate the behavior from before the user removal.