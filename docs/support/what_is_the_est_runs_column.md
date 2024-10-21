---
title: "What is the "Est. Runs" column?"
tags:
   - sweeps
---

W&B provides an estimated number of Runs that will occur when you create a W&B Sweep with a discrete search space. The total number of Runs is the cartesian product of the search space.

For example, suppose you provide the following search space:

![](/images/sweeps/sweeps_faq_whatisestruns_1.png)

The cartesian product in this example is 9. W&B shows this number in the W&B App UI as the estimated run count (**Est. Runs**):

![](/images/sweeps/spaces_sweeps_faq_whatisestruns_2.webp)


You can obtain the estimated Run count with the W&B SDK as well. Use the Sweep object's `expected_run_count` attribute to obtain the estimated Run count:

```python
sweep_id = wandb.sweep(
    sweep_configs, project="your_project_name", entity="your_entity_name"
)
api = wandb.Api()
sweep = api.sweep(f"your_entity_name/your_project_name/sweeps/{sweep_id}")
print(f"EXPECTED RUN COUNT = {sweep.expected_run_count}")
```