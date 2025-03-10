---
title: "Using artifacts with multiple architectures and runs?"
toc_hide: true
type: docs
tags:
   - artifacts
---
There are various methods to version a model. Artifacts provide a tool for model versioning tailored to specific needs. A common approach for projects that explore multiple model architectures involves separating artifacts by architecture. Consider the following steps:

1. Create a new artifact for each distinct model architecture. Use the `metadata` attribute of artifacts to provide detailed descriptions of the architecture, similar to the use of `config` for a run.
2. For each model, log checkpoints periodically with `log_artifact`. W&B builds a history of these checkpoints, labeling the most recent one with the `latest` alias. Refer to the latest checkpoint for any model architecture using `architecture-name:latest`.

