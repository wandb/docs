---
title: Weights and Biases の Anaconda パッケージはありますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- パイソン
---

Anaconda パッケージは `pip` または `conda` のどちらでもインストールできます。`conda` を使う場合は、 [conda-forge](https://conda-forge.org) チャンネルからパッケージを取得してください。

{{< tabpane text=true >}}
{{% tab "pip" %}}
```shell
# conda 環境を作成
conda create -n wandb-env python=3.8 anaconda
# 環境を有効化
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

インストール時に問題が発生した場合は、Anaconda の [パッケージ管理に関するドキュメント](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) を参照してください。