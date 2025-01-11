---
description: How to use the W&B platform to manage tables.
displayed_sidebar: default
---

# Manage Tables

Once you've [created a table](./tables-create.md), you can use the W&B platform to analyze, customize, and manage your data. 

## The W&B Platform
When you first open your workspace, you will be presented with a simple visualization of your tables. Each table you create appears in a `panel` where you can further customize your data.

You'll also notice a statement at the top of your panel. If you're following one of our example guides, it will look something like `runs.summary["Table Name"]`. You can edit this statement, which will be discussed later. 

In a default scenario, your table will display all of your runs, with each run having its own color so you can easily differentiate them.

### Format Tables


## Run Commands
Clicking on the funnel icon will bring up the filter menu. This will allow you to run various commands to filter your data. For example, if you wanted to only show columns that were alphanumeric, you could write something like `row["a"].isAlnum`.

Currently, you can filter by:
- index
- project
- row
- range

For more information on visualizing tables, see our [full guide](./visualize-tables.md).

## Export Data
You can easily export your table data to a .csv file by clicking the `Export to CSV` button, usually located at the bottom of a table's panel. Clicking this button automatically downloads a .csv file of your current table, with any formats or editing you have made.

For more methods of exporting data, see our [full guide](./tables-download.md).

## Next Steps
For a more in-depth walkthrough of how to use tables, see our [walkthrough](tables-walkthrough.md).