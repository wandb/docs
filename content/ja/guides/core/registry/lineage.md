---
title: リネージマップの作成と表示
description: W&B Registry でリネージ マップを作成します。
menu:
  default:
    identifier: ja-guides-core-registry-lineage
    parent: registry
weight: 8
---

W&B Registry のコレクション内では、ML 実験で使用された Artifacts の履歴を確認できます。この履歴は「リネージグラフ」と呼ばれます。

{{% pageinfo color="info" %}}
また、コレクションに含まれていない Artifacts についても、W&B にログした場合はリネージグラフで確認できます。
{{% /pageinfo %}}

リネージグラフでは、どの run が artifact をログしたかを特定できます。さらに、どの run がある artifact を入力として使用したかも表示されます。つまり、リネージグラフを使うことで、run の入力と出力のどちらも可視化できます。

たとえば、次の図は ML 実験全体で作成および使用された Artifacts を表しています。

{{< img src="/images/registry/registry_lineage.png" alt="Registry lineage" >}}

図の左から右にかけて、次の流れが示されています。
1. 複数の run が `split_zoo_dataset:v4` artifact をログしています。
2. "rural-feather-20" という run が、`split_zoo_dataset:v4` artifact をトレーニング用に利用しています。
3. "rural-feather-20" run の出力として `zoo-ylbchv20:v0` というモデル artifact が生成されます。
4. "northern-lake-21" run がこのモデル artifact `zoo-ylbchv20:v0` を使ってモデルを評価しています。


## run の入力をトラッキングする

artifact を run の入力または依存関係としてマークするには、`wandb.init.use_artifact` API を使用します。

以下のコードスニペットは `use_artifact` の使い方を示しています。山括弧（`< >`）で囲まれた値はご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# artifact を取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```


## run の出力をトラッキングする

run の出力として artifact を宣言するには、([`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}})) を利用します。

次のコードスニペットは `wandb.init.log_artifact` API の使い方です。山括弧（`< >`）で囲まれた値は必ずご自身の値に置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# run の出力として artifact をログ
run.log_artifact(artifact_or_path = artifact)
```

Artifacts の作成方法については、[アーティファクトの作成]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) をご参照ください。


## コレクション内でリネージグラフを表示する

W&B Registry のコレクションに紐づく artifact のリネージを確認できます。

1. W&B Registry にアクセスします。
2. Artifact を含むコレクションを選択します。
3. ドロップダウンから、リネージグラフを見たい artifact のバージョンをクリックします。
4. 「Lineage」タブを開きます。

artifact のリネージグラフページに移動したら、そのグラフ上の任意のノードに関する追加情報を確認できます。

run ノードを選択すると、その run の詳細（ID、名前、状態など）を確認できます。例として、次の図は `rural-feather-20` run の情報を表示しています。

{{< img src="/images/registry/lineage_expanded_node.png" alt="Expanded lineage node" >}}

artifact ノードを選択すると、その artifact の詳細（フルネーム、型、作成時刻、紐付けられたエイリアスなど）を表示します。

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="Expanded artifact node details" >}}