---
description: 巻き戻し
displayed_sidebar: default
---


# Rewind Runs
:::caution
runを巻き戻す機能はプライベートプレビュー中です。この機能にアクセスするためには、support@wandb.com までW&Bサポートにお問い合わせください。
:::

`resume_from` を [`wandb.init()`](https://docs.wandb.ai/ref/python/init) と一緒に使用して、runの履歴を特定のステップまで「巻き戻す」ことができます。runを巻き戻すと、W&Bは指定されたステップまでrunの状態をリセットし、元のデータを保持しつつ、一貫したrun IDを維持します。この機能により、元データを失うことなく、run履歴の修正や変更が可能になり、その時点から新しいデータをログできます。要約メトリクスは、新しくログされた履歴に基づいて再計算されます。

:::info
runを巻き戻すには、単調に増加するステップが必要です。[`define_metric()`](https://docs.wandb.ai/ref/python/run#define_metric) で定義された非単調ステップを使用してフォークポイントを設定することはできません。これにより、run履歴とシステムメトリクスの基本的な時系列順序が乱れてしまうためです。
:::

:::info
runを巻き戻すには、[`wandb`](https://pypi.org/project/wandb/) SDK バージョン >= 0.17.1 が必要です。
:::

### History and Config Management

- **履歴の切り捨て**: 履歴は巻き戻しポイントまで切り捨てられ、新しいデータログが可能になります。
- **要約メトリクス**: 新しくログされた履歴に基づいて再計算されます。
- **設定の保持**: 元の設定は保持され、新しい設定と統合することができます。

### Run Management

- **Runのアーカイブ**: 元のrunはアーカイブされ、[**`Run Overview`**](https://docs.wandb.ai/guides/app/pages/run-page#overview-tab)からアクセスできます。
- **Artifactの継承**: 新しいrunは元のrunからArtifactsを継承します。
- **Artifactの関連付け**: Artifactsは巻き戻されたrunの最新バージョンに関連付けられます。
- **不変のrun ID**: 正確な状態から一貫して巻き戻すために導入されました。
- **不変のrun IDのコピー**: run管理を改善するために、不変のrun IDをコピーするボタンがあります。

### Rewind and Forking Integration

RewindはForking機能を補完し、runの管理と実験により柔軟性を提供します。Forkingが特定のポイントから新しいブランチを作成して異なるパラメータやモデルを試すのに対し、Rewindはrun履歴自体を修正または変更することができます。

### Rewind a Run

runを巻き戻すには、`wandb.init()`の `resume_from` 引数を使用し、巻き戻したいrunの名前とステップを指定します。

```python
import wandb

# 初期化とデータログ
run = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(1000):
    run.log({"metric": i**2})
run.finish()

# runをステップ500に巻き戻す
rewind_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    resume_from="your_run_name",
    step=500
)

# ステップ500以降の新しいデータをログ
for i in range(500, 1000):
    rewind_run.log({"metric": i*2})
rewind_run.finish()
```
### Fork from a Rewound Run

巻き戻したrunからフォークするには、`wandb.init()`の `fork_from` 引数を使用し、ソースrun IDとフォークするステップを指定します。

```python 
import wandb

# 特定のステップからrunをフォーク
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 新しいrunでログを続行
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```

### Unsupported Functionality
- **ログの巻き戻し**: ログは新しいrunセグメントでリセットされます。
- **システムメトリクスの巻き戻し**: 巻き戻しポイント以降の新しいシステムメトリクスのみがログされます。
- **特定のrunセグメントに関連付けられたArtifacts**: Artifactsは、生成されたセグメントではなく最新のrunセグメントに関連付けられます。