---
description: Restricted Projects for collaborating on AI workflows with sensitive data
displayed_sidebar: default
---

# Restrict projects 

Define the scope of a W&B project grant access or limit who can view, edit, and submit W&B runs to your projects. Only the owner of the project, or a team admin, can edit or set a project's scope.

## Project scope types
There are four project scopes you can choose from. In order of most public to most private, they are: 
* _Open_: Anyone can submit runs or reports.
* _Public_: Anyone can view this project. Only your team can edit.
* _Team_: Only your team can view and edit this project.
* _Restricted_: Only invited members can view this project. Public sharing is disabled.

<!-- Each member inherits their team scoped role when you add them to your restricted project. -->

:::tip
Set a project's scope to **Restricted** if you who want to collaborate on workflows related to sensitive or confidential data. When you create a restricted project within a team, you can invite or add specific members from the team to collaborate on relevant experiments, artifacts, reports and so forth. 

Unlike other project scopes, all members of a team do not get implicit access to a restricted project. At the same time, team admins can join restricted projects to monitor team activity.
:::

## Set the scope on a new or existing project

Set a project's scope when you create a new project or after it is already created. 

:::info
* Only the owner of the project or a team admin can edit or set a project's scope.
* When a team admin enables **Make all future team projects private (public sharing not allowed)** within a team's privacy setting, that disables **Open** and **Public** project scopes for that team. In this case, your team can only use Team and Restricted scopes.
:::

:::tip
Talk to your admin or security team before you add or edit a project's scope.
:::

### Set project scope when you create a new project

1. Navigate to the W&B App at [https://wandb.ai/home](https://wandb.ai/home).
2. Click the **New Project** button in the upper right hand corner.
3. From the **Project Visibility dropdown**, select the desired visibility type.

Complete the following step if you select **Restricted** visibility. 

4. Provide one or more names of W&B Team members in the **Invite team members** field.
![](/images/hosting/restricted_project_2.png)
<!-- ![](/images/hosting/restricted_project_1.png) -->
### Set project scope to an existing project

1. Navigate to your W&B Project.
2. Select the **Overview** tab on the left column.
3. Click the **Edit Project Details** button on the upper right corner.  
4. From the **Project Visibility dropdown**, select the desired visibility type.

Complete the following step if you select **Restricted** visibility. 

5. Provide one or more names of W&B Team members in the **Invite team members** field.


## Edit a project's existing scope
1. Navigate to your W&B Project.
2. Select the **Overview** tab on the left column.
3. Click the **Edit Project Details** button on the upper right corner.  
4. From the **Project Visibility dropdown**, select the desired visibility type.

:::caution
All members of a team will have access to a project if you change the scope of a project from **Restricted** to **Team**.
:::

Complete the following step if you select **Restricted** visibility. 

5. Provide one or more names of W&B Team members in the **Invite team members** field.

:::caution
* All members of a team lose access to a project if you change the project's scope from **Team** to **Restricted**, unless you invite those team members to the project. 
* If you remove a team member when editing a restricted project, they do not have access to that project anymore.
:::

:::info
If you want to use a team-level service account in a restricted project, you should invite or add that specifically to the project. Otherwise a team-level service account can not access a restricted project by default.
:::

<!-- ## Editing a restricted project

If you are the owner of a restricted project you or a team admin can edit it by using the `Edit project details` button in the project overview tab, and add or remove team members from the opened interface. You can also change the scope of the project to _Team_ if needed.

 -->

<!-- ![](/images/hosting/restricted_project_edit.png) -->
## Other key things to note

* You can not move runs from a restricted project, but you can move runs from a non-restricted project to a restricted one.
* You can convert the visibility of a restricted project to only **Team** scope, irrespective of the team privacy setting **Make all future team projects private (public sharing not allowed)**.
* If the owner of a restricted project is not part of the parent team anymore, the team admin should change the owner to ensure seamless operations in the project.
