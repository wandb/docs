---
title: Rewind a run
description: 巻き戻し
menu:
  default:
    identifier: ja-guides-models-track-runs-rewind
    parent: what-are-runs
---

# Run を巻き戻す
{{% alert color="secondary" %}}
Run を巻き戻すオプションはプライベートプレビュー中です。この機能へのアクセスをリクエストするには、support@wandb.com の W&B サポートにお問い合わせください。

W&B は現在以下をサポートしていません:
* **ログの巻き戻し**: ログは新しい Run セグメントでリセットされます。
* **システムメトリクスの巻き戻し**: W&B は巻き戻しポイント後の新しいシステムメトリクスのみをログします。
* **アーティファクトの関連付け**: W&B はアーティファクトを生成するソースの Run に関連付けます。
{{% /alert %}}

{{% alert %}}
* Run を巻き戻すには、[W&B Python SDK](https://pypi.org/project/wandb/) バージョンが `0.17.1` 以上である必要があります。
* 単調増加するステップを使用する必要があります。実行履歴とシステムメトリクスの時系列の順序が必要なため、[`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ja" >}}) で定義された非単調ステップを使用することはできません。
{{% /alert %}}

Run を巻き戻して、元のデータを失うことなく Run の履歴を修正または変更します。さらに、Run を巻き戻すと、その時点から新しいデータをログできます。W&B は、巻き戻された Run のサマリーメトリクスを新たにログされた履歴に基づいて再計算します。これは以下の振る舞いを意味します:
- **履歴の切り捨て**: W&B は履歴を巻き戻しポイントまで切り捨て、新たなデータのログを可能にします。
- **サマリーメトリクス**: 新たにログされた履歴に基づいて再計算されます。
- **設定の保存**: W&B は元の設定を保存し、新しい設定をマージできます。

Run を巻き戻すと、W&B は指定されたステップに Run の状態をリセットし、元のデータを保存し、一貫した Run ID を保持します。これは次のことを意味します:

- **Run のアーカイブ**: W&B は元の Run をアーカイブします。Run は [**Run Overview**]({{< relref path="./#overview-tab" lang="ja" >}}) タブからアクセス可能です。
- **アーティファクトの関連付け**: アーティファクトを生成する Run に関連付けます。
- **不変の Run ID**: 正確な状態からの一貫したフォークのために導入されました。
- **不変の Run ID をコピー**: 改善された Run 管理のための不変の Run ID をコピーするボタン。

{{% alert title="Rewind とフォーキングの互換性" %}}
フォーキングは巻き戻しを補完します。

Run からフォークする際、W&B は特定のポイントで Run から新しいブランチを作成し、異なるパラメータやモデルを試すことができます。

Run を巻き戻す際、W&B は Run の履歴自体を修正または変更できるようにします。
{{% /alert %}}

## Run を巻き戻す

`resume_from` を利用して、[`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}) で Run の履歴を特定のステップまで「巻き戻す」。巻き戻したい Run の名前とステップを指定します:

```python
import wandb
import math

# 最初の Run を初期化し、いくつかのメトリクスをログする
# your_project_name と your_entity_name を置き換えてください！
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 特定のステップで最初の Run から巻き戻し、ステップ 200 からメトリクスをログする
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しい Run でログを続ける
# 最初の数ステップでは、run1 からメトリクスをそのままログする
# ステップ 250 以降はスパイキーパターンをログし始める
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # スパイクなしに run1 からのログを続行
    else:
        # ステップ 250 からスパイキーパターンを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーパターンを適用
        run2.log({"metric": subtle_spike, "step": i})
    # さらにすべてのステップで新しいメトリクスをログ
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブされた Run を表示する

Run を巻き戻した後、W&B アプリ UI を使用してアーカイブされた Run を探索できます。以下の手順でアーカイブされた Run を表示します:

1. **Overview タブにアクセスする:** Run のページで [**Overview タブ**]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。このタブは Run の詳細と履歴について包括的なビューを提供します。
2. **Forked From フィールドの位置を確認する:** **Overview** タブ内で `Forked From` フィールドを見つけます。このフィールドは再開の履歴をキャプチャします。**Forked From** フィールドには、ソース Run へのリンクが含まれており、元の Run にトレースバックして、巻き戻し履歴全体を理解することができます。

`Forked From` フィールドを使用して、アーカイブされた再開のツリーを簡単にナビゲートし、各巻き戻しのシーケンスと起源についての洞察を得ることができます。

## 巻き戻した Run からフォークする

巻き戻した Run からフォークするには、`wandb.init()` の [**`fork_from`**]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}}) 引数を使用し、ソース Run ID とフォークするソース Run からのステップを指定します:

```python 
import wandb

# 特定のステップから Run をフォークする
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しい Run でログを続ける
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```