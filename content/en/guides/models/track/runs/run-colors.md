---
description: 
menu:
  default:
    identifier: run-colors
    parent: Customize run colors
title: Customize run colors
---

W&B automatically assigns a color to each run that you create in your project. You can change the default color of a run to help you visually distinguish it from other runs in the table and graphs. Reset your project workspace to restore the default colors for all runs in the table.

Run colors are locally scoped. On the project page, custom colors apply only to your own workspace. In reports, custom colors for runs apply only at the section level. You can visualize the same run in different sections, which can use different custom colors per section.

## Edit default run colors

1. Click the **Runs** tab from the project sidebar.
2. Click the dot color next to the run name in the **Name** column.
3. Select a color from the color palette or the color picker, or enter a hex code.

{{< img src="/images/runs/run-color-palette.png" alt="Edit default run color in project workspace">}}

## Randomize run colors

To randomize the colors of all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Randomize run colors** from the dropdown menu.

{{% alert %}}
The option to randomize run colors is available only after modify the run's table in some way, such as by sorting, filtering, searching, or grouping.
{{% /alert %}}


## Reset run colors

<!-- {{% alert %}}
The option to randomize run colors is only available if there are at least two runs in the table or selector, and you have made some kind of modification to the view (sorting, filtering, searching, or grouping).
{{% /alert %}} -->

To restore the default colors for all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Reset colors** from the dropdown menu.

{{< img src="/images/runs/reset-run-colors.png" alt="Reset run colors in project workspace">}}

## Colorblind-safe palettes

W&B now offers two colorblind-safe color palettes to improve accessibility and ensure that your experiments are easily distinguishable by all team members, including those with color vision deficiency.

### Enabling colorblind-safe palettes

To use the colorblind-safe palettes in your workspace:

1. Navigate to your W&B project
2. Click on the **Workspace** tab from the project sidebar
3. Click the **Settings** icon (⚙️) in the top right corner
4. Select **Runs** from the settings drawer
5. In the **Color Palette** section, choose one of the two colorblind-safe palettes

### Available colorblind-safe palettes

W&B provides two carefully designed colorblind-safe palettes:

- **Colorblind-Safe Palette 1**: Optimized for deuteranopia (red-green color blindness), the most common form of color vision deficiency
- **Colorblind-Safe Palette 2**: Optimized for protanopia and tritanopia, providing maximum contrast for other forms of color blindness

Both palettes use colors that are distinguishable across different types of color vision deficiency while maintaining visual appeal for users with typical color vision.

### Benefits for accessibility

Using colorblind-safe palettes provides several important benefits:

- **Improved team collaboration**: Ensures all team members can effectively analyze and discuss experiment results, regardless of color vision abilities
- **Better visual distinction**: Colors are selected to maximize contrast and distinguishability, even on different monitor types and lighting conditions
- **Professional presentation**: Makes your dashboards and reports accessible for presentations and publications
- **Inclusive design**: Demonstrates commitment to accessibility, benefiting approximately 8% of men and 0.5% of women who have some form of color vision deficiency

The colorblind-safe palettes work seamlessly with all W&B visualization features, including line plots, scatter plots, parallel coordinates plots, and run tables.