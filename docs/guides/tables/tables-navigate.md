---
description: Manage tables with W&B.
displayed_sidebar: default
---

# Manage tables

Once you [create a table](./tables-create.md), you can use the W&B platform to analyze, customize, and manage you data. 

## The W&B platform
When first opening your workspace, you can see  a simple visualization of your tables. Each table you create appears in a `panel` where you can further customize your data.

There is also a statement on the top of your panel. If you're following one of the example guides, this statement should bevalv `runs.summary["Table Name"]`. You can also edit this statement. 

In a default scenario, your table displays all of your runs, with each run having its own color so you can easily differentiate them.

## Run commands
Clicking on the funnel icon brings up the filter menu. This allows you to run various commands to filter your data. For example, if you wanted to only show columns that were alphanumeric, you could write something like `row["a"].isAlnum`.

Currently, you can filter by:
- index
- project
- row
- range

For more information on visualizing tables, see the [full guide](./visualize-tables.md)

## Export data
Easily export your table data to a .csv file by clicking the `Export to CSV` button, usually located at the bottom of a table's panel. Clicking this button automatically downloads a .csv file of your current table, with any formats or editing you have made.

For more methods of exporting data, see the [full guide](./tables-download.md).

## Next steps
For a more in-depth walkthrough of how to use tables, see the [walkthrough](tables-walkthrough.md).