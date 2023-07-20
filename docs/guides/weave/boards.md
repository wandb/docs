---
slug: /guides/weave/boards
description: Boards are interconnected groups of Panels
displayed_sidebar: default
---

# Boards 

![](/images/weave/weave_latest_board.png)

Dynamically visualize, configure, and extend any StreamTable as a Weave Board. A Weave Board is a fully-customizable arrangement of Weave Panels and their underlying data, with versioned history. From the Weave Board UI, you can iteratively edit Weave Panels, load in any existing W&B Tables, create and link new Panels, make charts (scatter, line, bar, etc), define and call external APIs, and much more as your project evolves.

## Boards support an exploratory, branching workflow

1. Seed new Boards from any Weave Panel and rename them for easy tracking and future reference.
2. Customize your Board in the UI—any changes in the current session are automatically saved in a new working branch.
3. Click "commit" in the top right whenever you want to commit the current session/latest sequence of local changes to the current Board name.
4. Click "publish" in the top right to share your named Board to [weave.wandb.ai](https://weave.wandb.ai).

## Browse Tables and Boards from Weave Home 

![](/images/weave/weave_home.png)

Go to the Weave Home page at [weave.wandb.ai](https://weave.wandb.ai) to see all of your Tables and Boards stored in W&B. You can browse by entity (username or team) from the left sidebar and scroll through projects by name. W&B public cloud projects will display up to the 100 most recent Tables logged to that project by name. Currently, all Weave Boards for a given entity will be saved in the "weave" project and follow the privacy settings of that project in the W&B cloud.

## Seed a Board

You can create an empty Board via "+ New board" on the Weave Home page, or seed a Weave Board with an existing Weave Panel:

### From Weave Home

1. Log in to W&B
2. Go to the Weave Home page at [weave.wandb.ai](https://weave.wandb.ai). 
3. Navigate to the relevant project and Table you've logged to W&B (e.g. find the StreamTable by name)
4. Preview the Table and click "Seed new board".

![](/images/weave/seed_new_board.gif)

### From a local Jupyter notebook

1. Starting from a Weave Panel rendered in a notebook cell, open the Panel in a new tab to show a full-screen dashboard. 
![](/images/weave/stream_table_from_notebook.png)
2. Click “Seed new board” in the bottom right.