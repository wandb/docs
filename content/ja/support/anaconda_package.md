---
title: Is there an anaconda package for Weights and Biases?
menu:
  support:
    identifier: ja-support-anaconda_package
tags:
- python
toc_hide: true
type: docs
---

`pip` または `conda` のいずれかを使用してインストールできる anaconda パッケージがあります。 `conda` の場合は、[conda-forge](https://conda-forge.org) チャンネル からパッケージを入手してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境 を作成
conda create -n wandb-env python=3.8 anaconda
# 環境 をアクティブ化
conda activate wandb-env
# pip を使用して wandb をインストール
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

インストールに関する問題については、Anaconda の [パッケージ管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) を参照してください。
