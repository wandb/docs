---
title: run の メトリクスを比較
description: 複数の run のメトリクスを比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-run-comparer
    parent: panels
weight: 70
---

Run Comparer を使って、プロジェクト内の run 同士の違いと共通点を確認します。

## Run Comparer パネルを追加する

1. ページ右上の **Add panels** ボタンを選択します。
1. **Evaluation** セクションから **Run comparer** を選択します。

## Run Comparer を使う
Run Comparer は、プロジェクトで表示されている最初の 10 個の run について、run ごとに 1 列で、設定 と ログ済みメトリクス を表示します。

- 比較する run を変更するには、左側の run 一覧で検索、フィルター、グループ化、並べ替えを行います。Run Comparer は自動的に更新されます。
- Run Comparer 上部の検索フィールドを使って、Python バージョンや run の作成時刻などの 設定 キーや メタデータ キーを検索・フィルタリングできます。
- 差分だけを素早く確認し同一の 値 を隠すには、パネル上部の **Diff only** を切り替えます。
- 列幅や行の高さを調整するには、パネル上部の書式ボタンを使用します。
- 設定 または メトリクス の 値 をコピーするには、その 値 にマウスを載せ、コピーボタンをクリックします。画面に収まりきらない場合でも、 値 全体がコピーされます。

{{% alert %}}
デフォルトでは、Run Comparer は [`job_type`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の 値 が異なる run を区別しません。つまり、プロジェクト内で本来は比較できない run 同士を比較できてしまう可能性があります。たとえば、トレーニング run と モデルの評価 run を比較できてしまいます。トレーニング run には、run のログ、ハイパーパラメーター、トレーニング損失のメトリクス、そしてモデル自体が含まれる場合があります。評価 run は、モデルを使って新しいトレーニングデータ上でのモデルの性能を確認するだけかもしれません。

Runs Table の run 一覧で検索、フィルター、グループ化、並べ替えを行うと、Run Comparer は先頭の 10 個の run を比較するよう自動で更新されます。似た run を比較するには、Runs Table 内でフィルターや検索を行ってください。たとえば `job_type` でリストをフィルターまたは並べ替えます。詳しくは、[run のフィルタリング]({{< relref path="/guides/models/track/runs/filter-runs.md" lang="ja" >}}) を参照してください。
{{% /alert %}}