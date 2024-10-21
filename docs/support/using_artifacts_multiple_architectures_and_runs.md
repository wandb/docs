---
title: "Using artifacts with multiple architectures and runs?"
tags:
   - artifacts
---

There are many ways in which you can think of _version_ a model. Artifacts provides a you a tool to implement model versioning as you see fit. One common pattern for projects that explore multiple model architectures over a number of runs is to separate artifacts by architecture. As an example, one could do the following:

1. Create a new artifact for each different model architecture. You can use `metadata` attribute of artifacts to describe the architecture in more detail (similar to how you would use `config` for a run).
2. For each model, periodically log checkpoints with `log_artifact`. W&B will automatically build a history of those checkpoints, annotating the most recent checkpoint with the `latest` alias so you can refer to the latest checkpoint for any given model architecture using `architecture-name:latest`

## Reference Artifact FAQs