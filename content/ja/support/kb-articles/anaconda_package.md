---
title: Is there an anaconda package for Weights and Biases?
menu:
  support:
    identifier: ja-support-kb-articles-anaconda_package
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

`pip`または`conda`のどちらかを使用してインストールできるanacondaパッケージがあります。`conda`の場合、[conda-forge](https://conda-forge.org) チャンネルからパッケージを入手してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境を作成
conda create -n wandb-env python=3.8 anaconda
# 環境をアクティブ化
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

インストールに関する問題については、Anacondaの[パッケージの管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)を参照してください。