---
displayed_sidebar: default
title: Move runs
---

## Move runs between your projects

To move runs from one project to another:

1. Expand the table
2. Click the checkbox next to the runs you want to move
3. Click move and select the destination project

![](/images/app_ui/howto_move_runs.gif)



## Move runs to a team

On the project page:

1. Click the table tab to expand the runs table
2. Click the checkbox to select all runs
3. Click **Move**: the destination project can be in your personal account or any team that you're a member of.

![](/images/app_ui/demo_move_runs.gif)

## Send new runs to a team

In your script, set the entity to your team. "Entity" just means your username or team name. Create an entity (personal account or team account) in the web app before sending runs there.

```python
wandb.init(entity="example-team")
```

Your **default entity** is updated when you join a team. This means that on your [settings page](https://app.wandb.ai/settings), you'll see that the default location to create a new project is now the team you've just joined. Here's an example of what that [settings page](https://app.wandb.ai/settings) section looks like:

![](/images/app_ui/send_new_runs_to_team.png)
