---
description: How to use the sidebar and table on the project page
menu:
  default:
    identifier: filter-runs
    parent: what-are-runs
title: Filter and search runs
---

Use your project page to gain insights from runs logged to W&B.

## Filter runs

Filter runs based on their status, tags, or other properties with the filter button.


### Filter runs with tags

Filter runs based on their tags with the filter button.

{{< img src="/images/app_ui/filter_runs.gif" alt="" >}}


### Filter runs with regex

If regex doesn't provide you the desired results, you can make use of [tags](tags.md) to filter out the runs in Runs Table. Tags can be added either on run creation or after they're finished. Once the tags are added to a run, you can add a tag filter as shown in the gif below.

{{< img src="/images/app_ui/tags.gif" alt="If regex doesn't provide you the desired results, you can make use of tags to filter out the runs in Runs Table" >}}




## Search run names

Use [regex](https://dev.mysql.com/doc/refman/8.0/en/regexp.html) to find runs with the regex you specify. When you type a query in the search box, that will filter down the visible runs in the graphs on the workspace as well as filtering the rows of the table.



## Sort runs by minimum and maximum values
Sort the runs table by the minimum or maximum value of a logged metric. This is particularly useful if you want to view the best (or worst) recorded value.

The following steps describe how to sort the run table by a specific metric based on the minimum or maximum recorded value:

1. Hover your mouse over the column with the metric you want to sort with.
2. Select the kebob menu (three vertical lines).
3. From the dropdown, select either **Show min** or **Show max**.
4. From the same dropdown, select **Sort by asc** or **Sort by desc** to sort in ascending or descending order, respectively. 

{{< img src="/images/app_ui/runs_min_max.gif" alt="" >}}

## Search End Time for runs

We provide a column named `End Time` that logs that last heartbeat from the client process. The field is hidden by default.

{{< img src="/images/app_ui/search_run_endtime.png" alt="" >}}





## Export runs table to CSV

Export the table of all your runs, hyperparameters, and summary metrics to a CSV with the download button.

{{< img src="/images/app_ui/export_to_csv.gif" alt="" >}}


<!-- ## Edit run colors

When a new run is created, it is assigned a default color. You can edit the color for a given run by clicking the color preview.

Colors are locally scoped. On the project page, custom colors only apply to your own workspace. In reports, custom colors for runs only apply at the section level. You can visualize the same run in different sections, and it can have a different custom color in each section.

1. Select the Run you want to visualize
2. Click the colored dot 
3. Select a color for the graphs of your run
## See active runs

Look for a green dot next to the name of runs— this indicates they're active in the table and on the graph legends. -->

<!-- ## Bulk select runs

Delete multiple runs at once, or tag a group of runs— bulk selection makes it easier to keep the runs table organized.

{{< img src="/images/app_ui/howto_bulk_select.gif" alt="" >}} -->

<!-- ## Select all runs in table

Click the checkbox in the upper left corner of the table, and click "Select all runs" to select every run that matches the current set of filters.

{{< img src="/images/app_ui/all_runs_select.gif" alt="" >}} -->

<!-- 
## Search columns in the table

Search for the columns in the table UI guide with the **Columns** button.

{{< img src="/images/app_ui/search_columns.gif" alt="" >}} -->