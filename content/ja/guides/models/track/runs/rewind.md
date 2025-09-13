---
title: run を巻き戻す
description: 巻き戻し
menu:
  default:
    identifier: ja-guides-models-track-runs-rewind
    parent: what-are-runs
---

# run を巻き戻す
{{% alert color="secondary" %}}
run を巻き戻すオプションはプライベートプレビュー中です。この機能の アクセス をリクエストするには、support@wandb.com まで W&B Support にご連絡ください。

W&B が現在サポートしていない点:
* ** ログの巻き戻し **: 新しい run セグメントではログがリセットされます。
* ** システムメトリクスの巻き戻し **: 巻き戻しポイント以降の新しいシステムメトリクスのみを W&B がログします。
* ** Artifact の関連付け **: W&B は Artifact を、それを生成したソース run に関連付けます。
{{% /alert %}}

{{% alert %}}
* run を巻き戻すには、[W&B Python SDK](https://pypi.org/project/wandb/) のバージョンが `0.17.1` 以上である必要があります。
* step は単調増加でなければなりません。[`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) で定義した非単調な step では、必要な run の履歴やシステムメトリクスの時系列順が崩れるため機能しません。
{{% /alert %}}

run の元データを失うことなく、run の履歴を修正・変更するために run を巻き戻せます。さらに、run を巻き戻した時点から新しい データ をログできます。W&B は、巻き戻した run のサマリーメトリクスを新しくログした履歴に基づいて再計算します。つまり、次のように振る舞います:
- ** 履歴のトランケーション **: W&B は履歴を巻き戻しポイントまで切り詰め、新しい データ のログを可能にします。
- ** サマリーメトリクス **: 新しくログした履歴に基づいて再計算されます。
- ** 設定の保持 **: W&B は元の設定を保持し、新しい設定をマージできます。

run を巻き戻すと、W&B は元の データ を保持しつつ、指定した step の状態に run をリセットし、一貫した run ID を維持します。これは次を意味します:

- ** Run のアーカイブ **: W&B は元の run をアーカイブします。[Overview]({{< relref path="./#overview-tab" lang="ja" >}}) タブから run に アクセス できます。
- ** Artifact の関連付け **: Artifact は、それを生成した run に関連付けられます。
- ** 不変の run ID **: 正確な状態から一貫してフォークできるよう導入されます。
- ** 不変の run ID をコピー **: run 管理を改善するために、不変の run ID をコピーするボタンが用意されています。

{{% alert title="Rewind と Fork の互換性" %}}
Fork は Rewind を補完します。

run から Fork すると、W&B は特定のポイントから新しいブランチを作成し、異なるパラメータやモデルを試すことができます。

run を Rewind すると、W&B は run の履歴自体を修正・変更できるようにします。
{{% /alert %}}

## run を巻き戻す

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) の `resume_from` を使って、特定の step まで run の履歴を「巻き戻し」ます。対象の run 名と、どの step から巻き戻すかを指定します:

```python
import wandb
import math

# 最初の run を初期化してメトリクスをログする
# your_project_name と your_entity_name を置き換えてください!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 最初の run を特定の step から巻き戻し、step 200 からメトリクスのログを再開する
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 新しい run でログを継続
# 最初のいくつかの step では run1 と同じ値をログ
# step 250 以降はスパイキーなパターンをログする
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # スパイクなしで run1 から継続してログ
    else:
        # step 250 からスパイキーな振る舞いを導入
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 微妙なスパイキーパターンを適用
        run2.log({"metric": subtle_spike, "step": i})
    # さらに全ての step で新しいメトリクスをログ
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## アーカイブ済み run を表示する

run を巻き戻した後、W&B App UI でアーカイブ済みの run を確認できます。次の手順でアーカイブ済み run を表示します:

1. ** Overview タブに アクセス **: run のページの [**Overview** タブ]({{< relref path="./#overview-tab" lang="ja" >}}) に移動します。このタブでは run の詳細と履歴を包括的に確認できます。
2. ** Forked From フィールドを見つける **: **Overview** タブ内で `Forked From` フィールドを探します。このフィールドは再開の履歴を保持します。**Forked From** フィールドにはソース run へのリンクが含まれており、元の run にさかのぼって全体の Rewind 履歴を把握できます。

`Forked From` フィールドを使うと、アーカイブされた再開の ツリー を簡単にたどり、各 Rewind の順序と起点を理解できます。

## 巻き戻した run からフォークする

巻き戻した run からフォークするには、`wandb.init()` の [`fork_from`]({{< relref path="/guides/models/track/runs/forking" lang="ja" >}}) 引数を使い、ソース run の ID と、どの step からフォークするかを指定します:

```python 
import wandb

# 特定の step から run をフォーク
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しい run でログを継続
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```