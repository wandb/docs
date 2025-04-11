---
title: Weights and バイアス の anaconda パッケージはありますか？
menu:
  support:
    identifier: ja-support-kb-articles-anaconda_package
support:
- "{< tabpane >}  \n{% tab header=\"前書き\" %}  \nこのガイドでは、Data Science プロジェクトで wandb\
  \ を使用してメトリクスを追跡する方法を説明します。ここの情報は、ラン、アーティファクト、チーム、エンティティに関する理解を深めるのに役立ちます。  \n{%\
  \ /tab %}  \n{% tab header=\"ステップ 1\" %}  \n**wandb をインストールします**  \n\nwandb をインストールするには、以下のコマンドを実行します。\
  \  \n\n```shell  \npip install wandb  \n```  \nこれで ready to track!  \n{% /tab %}\
  \  \n{% tab header=\"ステップ 2\" %}  \n**ログイン**  \n\n次に、以下のコマンドで wandb にログインします。  \n\
  \n```shell  \nwandb login  \n```  \nGitHub または Google の資格情報を使用してログインできます。  \n{%\
  \ /tab %}  \n{% tab header=\"ステップ 3\" %}  \n**wandb に実験を記録する**  \n\n次のスクリプトを実行します。\
  \  \n\n```python  \nimport wandb  \n\n# wandb プロジェクトを初期化  \nwandb.init(project=\"\
  my-first-project\")  \n\n# ハイパーパラメータをログに記録  \nconfig = wandb.config  \nconfig.learning_rate\
  \ = 0.01  \n\n# メインのトレーニングループ  \nfor x in range(10):     \n    # メトリクスをログに記録する \
  \   \n    wandb.log({\"loss\": loss_value, \"epoch\": epoch})  \n```  \n\nこれにより、すべてのログはリアルタイムであなたのウェブベースのダッシュボードに表示されます。\
  \  \n{% /tab %}  \n{< /tabpane >}  "
toc_hide: true
type: docs
url: /support/:filename
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