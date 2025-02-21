---
title: Create and view lineage maps
description: W&B Registry で リネージ マップを作成します。
menu:
  default:
    identifier: ja-guides-models-registry-lineage
    parent: registry
weight: 8
---

W&B Registry内のコレクションでは、ML の実験で使用される Artifacts の履歴を表示できます。この履歴は _リネージグラフ_ と呼ばれます。

{{% pageinfo color="info" %}}
コレクションの一部ではない、W&B に記録した Artifacts のリネージグラフを表示することもできます。
{{% /pageinfo %}}

リネージグラフは、Artifacts を記録する特定の run を示すことができます。さらに、リネージグラフは、どの run が Artifacts を入力として使用したかも示すことができます。言い換えれば、リネージグラフは run の入力と出力を示すことができます。

たとえば、次の画像は、ML の実験全体で作成および使用された Artifacts を示しています。

{{< img src="/images/registry/registry_lineage.png" alt="" >}}

左から右へ、画像は以下を示しています。
1. 複数の run が `split_zoo_dataset:v4` アーティファクト を記録します。
2. "rural-feather-20" run は、トレーニングに `split_zoo_dataset:v4` アーティファクト を使用します。
3. "rural-feather-20" run の出力は、`zoo-ylbchv20:v0` というモデル アーティファクト です。
4. "northern-lake-21" という run は、モデル アーティファクト `zoo-ylbchv20:v0` を使用してモデルを評価します。

## run の入力を追跡する

`wandb.init.use_artifact` API を使用して、Artifacts を run の入力または依存関係としてマークします。

次の コードスニペット は、`use_artifact` の使用方法を示しています。山括弧 (`< >`) で囲まれた 値 を、自身の 値 に置き換えてください。

```python
import wandb

# run を初期化する
run = wandb.init(project="<project>", entity="<entity>")

# アーティファクト を取得し、依存関係としてマークする
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```

## run の出力を追跡する

([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}})) を使用して、Artifacts を run の出力として宣言します。

次の コードスニペット は、`wandb.init.log_artifact` API の使用方法を示しています。山括弧 (`< >`) で囲まれた 値 を必ず自身の 値 に置き換えてください。

```python
import wandb

# run を初期化する
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# アーティファクト を run の出力として記録する
run.log_artifact(artifact_or_path = artifact)
```

Artifacts の作成方法の詳細については、[Artifacts の作成]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ja" >}}) を参照してください。

## コレクション内のリネージグラフを表示する

W&B Registry でコレクションにリンクされた Artifacts のリネージを表示します。

1. W&B Registry に移動します。
2. Artifacts を含むコレクションを選択します。
3. ドロップダウンから、リネージグラフを表示する Artifacts の バージョン をクリックします。
4. 「リネージ」タブを選択します。

Artifacts のリネージグラフページに移動すると、そのリネージグラフ内の任意の ノード に関する追加情報を表示できます。

run の ID、run の名前、run の状態など、その run の詳細を表示するには、run ノード を選択します。例として、次の画像は `rural-feather-20` run に関する情報を示しています。

{{< img src="/images/registry/lineage_expanded_node.png" alt="" >}}

Artifacts の詳細（フルネーム、タイプ、作成時間、関連する エイリアス など）を表示するには、Artifacts ノード を選択します。

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="" >}}
