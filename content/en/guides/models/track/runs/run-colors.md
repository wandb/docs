---
description: 
menu:
  default:
    identifier: run-colors
    parent: Customize run colors
title: Customize run colors
---

W&B automatically assigns a color to each run that you create in your project. You can customize these colors to help you visually distinguish runs in tables and graphs. Run colors are locally scoped—on the project page, custom colors apply only to your own workspace. In reports, custom colors for runs apply only at the section level.

## Choose a color palette

W&B provides several predefined color palettes to suit different needs and preferences. You can select a palette that works best for your team and use case.

### How to select a color palette

1. Navigate to your W&B project
2. Click on the **Workspace** tab from the project sidebar
3. Click the **Settings** icon in the top right corner
4. Select **Runs** from the settings drawer
5. In the **Color palette** section, choose from the available options

### Available color palettes

W&B offers three color palette options:

- **Default**: The standard W&B color palette with a vibrant range of colors
- **Colorblind-safe (deuteranomaly)**: Optimized for red-green color blindness, the most common form affecting approximately 6% of males
- **Colorblind-safe (all other forms)**: Optimized for protanopia, tritanopia, and other forms of color vision deficiency

The colorblind-safe palettes use carefully selected colors that remain distinguishable across different types of color vision deficiency while maintaining visual appeal for users with typical color vision.

## Edit individual run colors

You can manually change the color of specific runs to customize your visualization further.

1. Click the **Runs** tab from the project sidebar.
2. Click the dot color next to the run name in the **Name** column.
3. Select a color from the color palette or the color picker, or enter a hex code.

{{< img src="/images/runs/run-color-palette.png" alt="Edit default run color in project workspace">}}

## Randomize run colors

To randomize the colors of all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Randomize run colors** from the dropdown menu.

{{% alert %}}
The option to randomize run colors is available only after modifying the run's table in some way, such as by sorting, filtering, searching, or grouping.
{{% /alert %}}

## Reset run colors

To restore the default colors for all runs in the table:

1. Click the **Runs** tab from the project sidebar.
2. Hover over the **Name** column header, click the three horizontal dots (**...**), and select **Reset colors** from the dropdown menu.

{{< img src="/images/runs/reset-run-colors.png" alt="Reset run colors in project workspace">}}

## Accessibility benefits

Using colorblind-safe palettes provides several important benefits:

- **Improved team collaboration**: Ensures all team members can effectively analyze and discuss experiment results, regardless of color vision abilities
- **Better visual distinction**: Colors are selected to maximize contrast and distinguishability, even on different monitor types and lighting conditions
- **Professional presentation**: Makes your dashboards and reports accessible for presentations and publications
- **Inclusive design**: Demonstrates commitment to accessibility, benefiting approximately 8% of men and 0.5% of women who have some form of color vision deficiency

The selected color palette applies to all W&B visualization features, including line plots, scatter plots, parallel coordinates plots, and run tables.

## Color options for grouped runs

When using the **Run colors** setting, you have two options:

- **Original project colors**: Uses the default color assignment for runs
- **Key-based colors**: Colors runs based on metric or configuration values (see [Semantic run plot legends]({{< relref "color-code-runs" >}}) for details)