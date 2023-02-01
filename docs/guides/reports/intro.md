---
slug: /guides/reports
description: Project management and collaboration tools for machine learning projects
---

# Collaborative Reports

Use W&B Reports to organize Runs, embed and automate visualizations, describe your findings, and share updates with collaborators.

Use Reports to organize, embed and automate visualizations, describe your findings, share updates with collaborators, and more.

:::info
Check out our [video demo](https://www.youtube.com/watch?v=2xeJIv\_K\_eI) of Reports, or read curated Reports in [W&B Fully Connected](http://wandb.me/fc).
:::

<!-- {% embed url="https://www.youtube.com/watch?v=2xeJIv_K_eI" %} -->

## Typical use cases for reports

1. **Collaboration**: Share findings with your colleagues.
2. **Work log**: Track what you've tried and plan next steps.
3. **Automated Visualizations**: Integrate model analysis into your model CI/CD pipeline using the Report API.

### Notes: Add a visualization with a quick summary

Capture an important observation, an idea for future work, or a milestone reached in the development of a project. All experiment runs in your report will link to their parameters, metrics, logs, and code, so you can save the full context of your work.

Jot down some text and pull in relevant charts to illustrate your insight ([comparing networks in distributed training →](https://wandb.ai/stacey/estuary/reports/When-Inception-ResNet-V2-is-too-slow--Vmlldzo3MDcxMA) )

![](/images/reports/notes_add_quick_summary.png)

Save the best examples from a complex code base for easy reference and future interaction (example: [LIDAR point clouds → ](https://wandb.ai/stacey/lyft/reports/LIDAR-Point-Clouds-of-Driving-Scenes--Vmlldzo2MzA5Mg))

![](/images/reports/notes_add_quick_summary_save_best_examples.png)

### Collaboration: Share findings with your colleagues

Explain how to get started with a project, share what you've observed so far, and synthesize the latest findings. Your colleagues can make suggestions or discuss details using comments on any panel or at the end of the report.

Include dynamic settings so that your colleagues can explore for themselves, get additional insights, and better plan their next steps. In this example, three types of experiments can be visualized independently, compared, or averaged ([SafeLife benchmark experiments →](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)).

![](/images/reports/intro_collaborate1.png)

![](/images/reports/intro_collaborate2.png)

Use sliders and configurable media panels to showcase a model's results or training progress (example → [StarGANv2](https://wandb.ai/stacey/stargan/reports/Cute-Animals-and-Post-Modern-Style-Transfer-StarGAN-v2-for-Multi-Domain-Image-Synthesis---VmlldzoxNzcwODQ))

![](/images/reports/intro_collaborate3.png)

![](/images/reports/intro_collaborate4.png)

### Work log: Track what you've tried and plan next steps

Write down your thoughts on experiments, your findings, and any gotchas and next steps as you work through a project, keeping everything organized in one place. This lets you "document" all the important pieces beyond your scripts (example: [tuning transformers →](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)).

![](/images/reports/intro_work_log_1.png)

Tell the story of a project, which you and others can reference later to understand how and why a model was developed (example → [The View from the Driver's Seat](https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ))

![](/images/reports/intro_work_log_2.png)

### [See the OpenAI case study →](https://bit.ly/wandb-learning-dexterity)

<!-- Once you have [experiments in W&B](../../quickstart.md), easily visualize results in reports. Here's a quick overview video. -->

<!-- {% embed url="https://www.youtube.com/watch?v=o2dOSIDDr1w" %} -->
