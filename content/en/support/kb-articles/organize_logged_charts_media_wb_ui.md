---
url: /support/:filename
title: "How can I organize my logged charts and media in the W&B UI?"
toc_hide: true
type: docs
support:
   - experiments
---
The `/` character separates logged panels in the W&B UI. By default, the segment of the logged item's name before the `/` defines a group of panels known as a "Panel Section."

```python
import wandb

with wandb.init() as run:

   run.log({"val/loss": 1.1, "val/acc": 0.3})
   run.log({"train/loss": 0.1, "train/acc": 0.94})
```

In the [Workspace]({{< relref "/guides/models/track/project-page.md#workspace-tab" >}}) settings, adjust the grouping of panels based on either the first segment or all segments separated by `/`.