---
title: Weights and Biases の Anaconda パッケージはありますか？
menu:
  support:
    identifier: ja-support-kb-articles-anaconda_package
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

`pip` または `conda` のいずれかでインストール可能な Anaconda パッケージがあります。`conda` の場合は、[conda-forge](https://conda-forge.org) チャンネルから入手してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境を作成する
conda create -n wandb-env python=3.8 anaconda
# 環境を有効化する
conda activate wandb-env
# pip で wandb をインストールする
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

インストールに関する問題がある場合は、Anaconda の [パッケージ管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) を参照してください。