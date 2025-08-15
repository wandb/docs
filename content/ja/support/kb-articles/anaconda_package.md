---
title: Weights and Biases の anaconda パッケージはありますか？
menu:
  support:
    identifier: ja-support-kb-articles-anaconda_package
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

Anaconda パッケージは `pip` または `conda` のいずれかでインストール可能です。`conda` を使う場合は、[conda-forge](https://conda-forge.org) チャンネルからパッケージを取得してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境を作成
conda create -n wandb-env python=3.8 anaconda
# 環境をアクティブ化
conda activate wandb-env
# pip を使って wandb をインストール
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

インストールに関する問題については、Anaconda の [パッケージ管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) をご参照ください。