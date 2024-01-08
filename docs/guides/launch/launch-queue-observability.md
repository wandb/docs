---
displayed_sidebar: default
---

# Queue observability (beta)

Launch provides an interactive dashboard for each queue, allowing ML engineers and MLOps teams to make effective use of their available hardware. 

By exploring this tab, you can see when the queue was in heavy use or idle, visualize what workloads were running, and spot inefficient jobs.  For deeper analysis, the page links to the W&B experiment tracking workspace and to external infrastructure monitoring providers like Datadog, NVIDIA Base Command, or cloud consoles.

:::info
To turn on this capability, toggle on the `Launch observability` feature flag in your user settings.

This feature requires W&B Weave, which is not available on some Customer-managed and Dedicated Cloud.  Contact your W&B representative to learn more.
:::

## Dashboard and plots

Each queue now has a `Monitor` tab.   When you click on the tab, W&B will gather data over the last 7 days of activity for this queue.  Time ranges, grouping and filters can be controlled from the left panel.  

The dashboard contains a number of plots answering common questions about performance and efficiency.

### Job status

![](/images/launch/launch_obs_jobstatus.png)

This plot shows how many jobs were running, pending, queued, or completed in each time interval.  

This can be useful for identifying periods of idleness in the queue--in the case of fixed resources (e.g. a DGX BasePod), that might suggest an opportunity to run lower-priority pre-emptible jobs such as sweeps.  Meanwhile, with cloud resources, recurring bursts might suggest an opportunity to save money by reserving resources for particular times.

Seeing a many `Queued` items might indicate opportunities to shift workloads to other queues, while a spike in failures can identify users who might need help with their job setup.

Select a range to show more details in the plot below, or Zoom to filter the entire page.

### Queued time

![](/images/launch/launch_obs_queuedtime.png)

This plot shows, for every time period, the amount of time in minutes that jobs were queued.  For example, if there were 10 jobs queued in that period, waiting for an average of 30 minutes each, the plot would show 300 minutes. 

This plot can be colored by the `Grouping` control in the left bar--which can be particularly helpful for identifying which users and jobs are feeling the pain of scarce queue capacity.

### Job runs

![](/images/launch/launch_obs_jobruns2.png)


This plot shows the start and end of every job executed in a time period, with distinct colors for each run.  This makes it easy to see at a glance what workloads the queue was processing at a given time.  

Use the Select tool in the bottom right of the panel to brush over jobs to see more details in the table below.



### CPU and GPU Usage

![](/images/launch/launch_obs_gpu.png)

This quartet of plots sheds light on the efficiency of job runs.  You can see, in particular, whether a job run has taken a long time on GPU (the x-axes) while using a low percentage of its cores or memory.  


### Errors

![](/images/launch/launch_obs_errors.png)

Lastly, the page shows the last errors to occur on the queue, so that MLOps teams can help unblock ML engineers faster.


## External links

The queue observability dashboard's view is consistent across all queue types, but in many cases, it can be useful to jump directly into the environment-specific monitors.  To make this easier, you can add links to those consoles right in the queue observability dashboard.

At the bottom of the page, click `Manage Links` to open a panel.  Add the full URL of the page you want, then a label.  Upon saving, a new link will appear under the External Links section, and clicking this link will take you to the external product.











