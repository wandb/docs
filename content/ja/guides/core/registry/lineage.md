---
title: リネージ マップの作成と表示
description: W&B レジストリでリネージマップを作成します。
menu:
  default:
    parent: registry
    identifier: lineage
weight: 8
---

W&B Registry のコレクション内では、機械学習実験で使用された artifacts の履歴を確認できます。この履歴は _リネージ グラフ_ と呼ばれます。

{{% pageinfo color="info" %}}
W&B にログした artifacts がコレクションの一部でなくても、リネージ グラフを確認することができます。
{{% /pageinfo %}}

リネージ グラフでは、artifact をログした特定の run を可視化できます。さらに、artifact を入力として利用した run も示されます。つまり、リネージ グラフを使うことで run の入力と出力の両方を確認できます。

例えば、次の画像は ML 実験で作成・利用された artifacts の様子を表しています。

{{< img src="/images/registry/registry_lineage.png" alt="Registry lineage" >}}

左から右にかけて、画像は次の流れを示しています。
1. 複数の run が `split_zoo_dataset:v4` artifact をログしています。
2. "rural-feather-20" という run が、`split_zoo_dataset:v4` artifact をトレーニングに使用しています。
3. "rural-feather-20" run の出力は `zoo-ylbchv20:v0` という model artifact です。
4. "northern-lake-21" という run が、model artifact `zoo-ylbchv20:v0` を使ってモデルの評価を行っています。


## run の入力をトラッキングする

artifact を run の入力または依存関係としてマークするには、`wandb.init.use_artifact` API を使用します。

以下のコードスニペットは、`use_artifact` の使い方を示しています。山かっこ（`< >`）で囲まれている値はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# artifact を取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```


## run の出力をトラッキングする

run の出力として artifact を宣言するには、([`wandb.init.log_artifact`]({{< relref "/ref/python/sdk/classes/run.md#log_artifact" >}})) を使います。

次のコードスニペットは、`wandb.init.log_artifact` API の使い方を示しています。山かっこ（`< >`）で囲まれた値は、ご自身の値に差し替えてご利用ください。

```python
import wandb

# run を初期化
run = wandb.init(entity  = "<entity>", project = "<project>")
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# run の出力として artifact をログ
run.log_artifact(artifact_or_path = artifact)
```

artifact の作成について詳しくは、[Create an artifact]({{< relref "guides/core/artifacts/construct-an-artifact.md" >}}) をご覧ください。


## コレクション内でリネージ グラフを表示する

W&B Registry のコレクションにリンクされている artifact のリネージを確認できます。

1. W&B Registry にアクセスします。
2. artifact を含むコレクションを選択します。
3. ドロップダウンから、リネージグラフを表示したい artifact バージョンをクリックします。
4. 「Lineage」タブを選択します。

artifact のリネージ グラフのページに入ると、そのリネージグラフ内のノードごとに追加情報を閲覧できます。

run ノードをクリックすると、その run の詳細（run の ID・名前・状態など）を確認できます。例えば、次の画像は `rural-feather-20` run に関する情報を表示しています。

{{< img src="/images/registry/lineage_expanded_node.png" alt="Expanded lineage node" >}}

artifact ノードを選択すると、その artifact の詳細（フルネーム・タイプ・作成時刻・関連エイリアス など）を表示できます。

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="Expanded artifact node details" >}}