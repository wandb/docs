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

Anaconda パッケージがあり、`pip` または `conda` を使用してインストール可能です。`conda` の場合は、[conda-forge](https://conda-forge.org) チャンネルからパッケージを取得します。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境を作成する
conda create -n wandb-env python=3.8 anaconda
# 環境をアクティブにする
conda activate wandb-env
# pip を使用して wandb をインストールする
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

インストールに関する問題については、パッケージ管理に関する Anaconda の [ドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) を参照してください。