---
title: Create an experiment
description: W&B の 実験 を作成します。
menu:
  default:
    identifier: ja-guides-models-track-launch
    parent: experiments
weight: 1
---

W&B Python SDK を使用して、 機械学習 の 実験 を追跡します。次に、インタラクティブな ダッシュボード で 結果 を確認するか、[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) でプログラムによる アクセス のために Python に データ をエクスポートできます。

この ガイド では、W&B の構成要素を使用して W&B の 実験 を作成する方法について説明します。

## W&B の 実験 を作成する方法

W&B の 実験 は、次の 4 つの ステップ で作成します。

1. [W&B Run を初期化する]({{< relref path="#initialize-a-wb-run" lang="ja" >}})
2. [ハイパー パラメーター の 辞書 をキャプチャする]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ja" >}})
3. [トレーニング ループ 内で メトリクス を ログ 記録する]({{< relref path="#log-metrics-inside-your-training-loop" lang="ja" >}})
4. [アーティファクト を W&B に ログ 記録する]({{< relref path="#log-an-artifact-to-wb" lang="ja" >}})

### W&B の run を初期化する
スクリプト 呼び出しの先頭で、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を呼び出して、W&B Run として データ を同期および ログ 記録するバックグラウンド プロセス を生成します。

次の コード スニペット は、「cat-classification」という名前の新しい W&B プロジェクト を作成する方法を示しています。「My first experiment」というメモを追加して、この run を識別しやすくしました。タグ「baseline」と「paper1」は、この run が将来の論文出版を目的とした ベースライン 実験 であることを思い出させるために含まれています。

```python
# W&B Python ライブラリをインポートする
import wandb

# 1. W&B Run を開始する
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
W&B を `wandb.init()` で初期化すると、[Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) オブジェクト が返されます。さらに、W&B は、すべての ログ と ファイル が保存され、W&B サーバー に非同期的に ストリーミング されるローカル ディレクトリー を作成します。

{{% alert %}}
注: run は、wandb.init() を呼び出すときに、その プロジェクト が既に存在する場合、既存の プロジェクト に追加されます。たとえば、「cat-classification」という プロジェクト が既にある場合、その プロジェクト は引き続き存在し、削除されません。代わりに、新しい run がその プロジェクト に追加されます。
{{% /alert %}}

### ハイパー パラメーター の 辞書 をキャプチャする
学習率やモデル タイプなどのハイパー パラメーター の 辞書 を保存します。config でキャプチャするモデル 設定は、 後で 結果 を整理してクエリするのに役立ちます。

```python
# 2. ハイパー パラメーター の 辞書 をキャプチャする
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
実験 の構成方法の詳細については、[実験 の構成]({{< relref path="./config.md" lang="ja" >}}) を参照してください。

### トレーニング ループ 内で メトリクス を ログ 記録する
各 `for` ループ (エポック) 中に メトリクス を ログ 記録します。精度と損失の値が計算され、[`wandb.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}) で W&B に ログ 記録されます。デフォルトでは、wandb.log を呼び出すと、履歴 オブジェクト に新しい ステップ が追加され、概要 オブジェクト が更新されます。

次の コード例 は、`wandb.log` で メトリクス を ログ 記録する方法を示しています。

{{% alert %}}
モードを設定して データ を取得する方法の詳細は省略されています。
{{% /alert %}}

```python
# モデルとデータの設定
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        # 3. トレーニング ループ 内でメトリクスをログ記録して視覚化する
        # モデルのパフォーマンス
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B で ログ 記録できるさまざまな データ タイプ の詳細については、[実験 中の ログ データ]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。

### アーティファクト を W&B に ログ 記録する
オプションで、W&B Artifact を ログ 記録します。Artifacts を使用すると、 データセット と モデル を簡単に バージョン管理 できます。
```python
wandb.log_artifact(model)
```
Artifacts の詳細については、[Artifacts のチャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。モデル の バージョン管理 の詳細については、[Model Management]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) を参照してください。

### まとめ
上記の コード スニペット を含む完全な スクリプト は、次のとおりです。
```python
# W&B Python ライブラリをインポートする
import wandb

# 1. W&B Run を開始する
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

# 2. ハイパー パラメーター の 辞書 をキャプチャする
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# モデルとデータの設定
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        # 3. トレーニング ループ 内でメトリクスをログ記録して視覚化する
        # モデルのパフォーマンス
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. アーティファクト を W&B に ログ 記録する
wandb.log_artifact(model)

# オプション: 最後にモデルを保存する
model.to_onnx()
wandb.save("model.onnx")
```

## 次の ステップ : 実験 を視覚化する
W&B ダッシュボード を、 機械学習 モデル の 結果 を整理して視覚化するための一元的な場所として使用します。数回クリックするだけで、[パラレル座標図]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[ パラメータ の重要性分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) などのリッチでインタラクティブなグラフを作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

実験 と特定の run の表示方法の詳細については、[実験 からの 結果 の視覚化]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を参照してください。

## ベストプラクティス
次に、 実験 を作成するときに検討すべき推奨 ガイドライン をいくつか示します。

1.  **Config**: ハイパー パラメーター 、 アーキテクチャー 、 データセット 、およびモデル を再現するために使用したいその他のものを追跡します。これらは列に表示されます。config 列を使用して、アプリで run を動的にグループ化、並べ替え、フィルタリングします。
2.  **Project**: プロジェクト は、一緒に比較できる 実験 の セット です。各 プロジェクト には専用の ダッシュボード ページがあり、さまざまなモデル バージョン を比較するために、さまざまな run のグループを簡単にオン/オフにできます。
3.  **Notes**: スクリプト から直接、簡単な コミット メッセージ を設定します。W&B アプリ の run の 概要 セクションでメモを編集および アクセス します。
4.  **Tags**: ベースライン run とお気に入りの run を識別します。タグを使用して run をフィルタリングできます。W&B アプリ の プロジェクト の ダッシュボード の 概要 セクションで、後でタグを編集できます。

次の コード スニペット は、上記の ベストプラクティス を使用して W&B の 実験 を定義する方法を示しています。

```python
import wandb

config = dict(
    learning_rate=0.01, momentum=0.2, architecture="CNN", dataset_id="cats-0192"
)

wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)
```

W&B の 実験 を定義する際に使用可能な パラメータ の詳細については、[API Reference Guide]({{< relref path="/ref/python/" lang="ja" >}}) の [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API ドキュメント を参照してください。
