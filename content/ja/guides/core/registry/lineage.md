---
title: リネージ マップを作成および表示する
description: W&B Registry でリネージ マップを作成する。
menu:
  default:
    identifier: ja-guides-core-registry-lineage
    parent: registry
weight: 8
---

W&B レジストリ内のコレクションでは、ML 実験が使用するアーティファクトの履歴を確認することができます。この履歴は _リネージグラフ_ と呼ばれます。

{{% pageinfo color="info" %}}
コレクションの一部ではないアーティファクトに対しても、W&Bにログを記録したリネージグラフを表示することができます。
{{% /pageinfo %}}

リネージグラフは、アーティファクトをログする特定の run を表示できます。さらに、リネージグラフはどの run がアーティファクトを入力として使用したかも表示できます。言い換えると、リネージグラフはrun の入力と出力を表示できます。

例えば、次の画像は ML 実験全体で作成および使用されたアーティファクトを示しています。

{{< img src="/images/registry/registry_lineage.png" alt="" >}}

左から右に、画像は以下を示しています。
1. 複数の runs が `split_zoo_dataset:v4` アーティファクトをログします。
2. "rural-feather-20" run は `split_zoo_dataset:v4` アーティファクトをトレーニング用に使用します。
3. "rural-feather-20" run の出力は `zoo-ylbchv20:v0` というモデルのアーティファクトです。
4. "northern-lake-21" という run はモデルを評価するために `zoo-ylbchv20:v0` モデルアーティファクトを使用します。

## run の入力をトラックする

`wandb.init.use_artifact` API を使用して、run の入力または依存関係としてアーティファクトをマークします。

次のコードスニペットは、`use_artifact` の使用方法を示しています。山括弧 (`< >`) で囲まれた値をあなたの値に置き換えてください。

```python
import wandb

# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")

# アーティファクトを取得し、依存関係としてマークする
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```

## run の出力をトラックする

作成したアーティファクトの出力を run の出力として宣言するには、([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}})) を使用します。

次のコードスニペットは、`wandb.init.log_artifact` API の使用方法を示しています。山括弧 (`< >`) で囲まれた値をあなたの値に置き換えるようにしてください。

```python
import wandb

# run を初期化する
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# アーティファクトをログとして run の出力にする
run.log_artifact(artifact_or_path = artifact)
```

アーティファクトの作成に関する詳細については、[Create an artifact]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。

## コレクション内のリネージグラフを表示する

W&B レジストリ内のコレクションにリンクされたアーティファクトのリネージを表示します。

1. W&B レジストリに移動します。
2. アーティファクトを含むコレクションを選択します。
3. ドロップダウンから、リネージグラフを表示したいアーティファクトのバージョンをクリックします。
4. 「Lineage」タブを選択します。

アーティファクトのリネージグラフのページに移動すると、そのリネージグラフ内の任意のノードに関する追加情報を表示できます。

run ノードを選択して、その run の詳細（run の ID、run の名前、run の状態など）を表示します。例として、次の画像は `rural-feather-20` run に関する情報を示しています。

{{< img src="/images/registry/lineage_expanded_node.png" alt="" >}}

アーティファクトノードを選択して、そのアーティファクトの詳細（完全な名前、タイプ、作成時間、関連するエイリアスなど）を表示します。

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="" >}}