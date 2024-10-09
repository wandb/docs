---
displayed_sidebar: default
title: Scatter plots
---

Use the scatter plot to compare multiple runs and visualize how your experiments are performing. We've added some customizable features:

1. Plot a line along the min, max, and average
2. Custom metadata tooltips
3. Control point colors 
4. Set axes ranges
5. Switch axes to log scale

Here’s an example of validation accuracy of different models over a couple of weeks of experimentation. The tooltip is customized to include the batch size and dropout as well as the values on the axes. There’s also a line plotting the running average of validation accuracy.  
[See a live example →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

![](/images/general/scatter-plots-1.png)

## Common Questions

### Is it possible to plot the max of a metric rather than plot step by step?

The best way to do this is to create a Scatter Plot of the metric, go into the Edit menu, and select Annotations. From there you can plot the running max of the values
