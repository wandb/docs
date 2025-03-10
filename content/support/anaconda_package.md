---
title: "Is there an anaconda package for Weights and Biases?"
toc_hide: true
type: docs
tags:
- python
---
There is an anaconda package that is installable using either `pip` or `conda`. For `conda`, obtain the package from the [conda-forge](https://conda-forge.org) channel.

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# Create a conda environment
conda create -n wandb-env python=3.8 anaconda
# Activate the environment
conda activate wandb-env
# Install wandb using pip
pip install wandb
```
{{% /tab %}}
{{% tab "conda" %}}
```shell
conda activate myenv
conda install wandb --channel conda-forge
```
{{% /tab %}}
{{< /tabpane >}}

For installation issues, refer to this Anaconda [documentation on managing packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) for assistance.