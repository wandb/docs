---
description: 
menu:
  default:
    identifier: run-colors
    parent: Customize run colors
title: Customize run colors
---

W&B automatically assigns a color to each run that you create in your project. You can change the default color of a run to help you visually distinguish it from other runs in the table and graphs. Resetting your project workspace will restore the default colors for all runs in the table.

Run colors are locally scoped. On the project page, custom colors only apply to your own workspace. In reports, custom colors for runs only apply at the section level. You can visualize the same run in different sections, and it can have a different custom color in each section.

## Edit default run colors

1. Click the **Runs** tab from the project sidebar.
2. Click the dot color next to the run name in the **Name** column.
3. Select a color from the color palette, the color picker, or enter a hex code.

{{< img src="/images/runs/run-color-palette.png" alt="Edit default run color in project workspace">}}

## Randomize run colors

To randomize the colors of all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Randomize run colors** from the dropdown menu.

{{% alert %}}
The option to randomize run colors is only available if you have made some kind of modification to the run's table (sorting, filtering, searching, or grouping).
{{% /alert %}}


## Reset run colors

<!-- {{% alert %}}
The option to randomize run colors is only available if there are at least two runs in the table or selector, and you have made some kind of modification to the view (sorting, filtering, searching, or grouping).
{{% /alert %}} -->

To restore the default colors for all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Reset colors** from the dropdown menu.

{{< img src="/images/runs/reset-run-colors.png" alt="Reset run colors in project workspace">}}