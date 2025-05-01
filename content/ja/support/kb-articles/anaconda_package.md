---
title: Weights and バイアス の anaconda パッケージはありますか？
menu:
  support:
    identifier: ja-support-kb-articles-anaconda_package
support:
  - python
toc_hide: true
type: docs
url: /ja/support/:filename
---
`pip` または `conda` を使用してインストール可能な Anaconda パッケージがあります。`conda` の場合、[conda-forge](https://conda-forge.org) チャンネルからパッケージを取得してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境の作成
conda create -n wandb-env python=3.8 anaconda
# 環境をアクティベートする
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

インストールの問題については、Anaconda の[パッケージを管理するためのドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)を参照してください。