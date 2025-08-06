---
title: run メトリクスを比較する
description: 複数の run 間でメトリクスを比較する
menu:
  default:
    identifier: run-comparer
    parent: panels
weight: 70
---

Run Comparer を使うと、プロジェクト内の複数の run を比較し、その違いや共通点を確認できます。

## Run Comparer パネルの追加

1. ページ右上の **Add panels** ボタンをクリックします。
1. **Evaluation** セクションから **Run comparer** を選択します。

## Run Comparer の使い方
Run Comparer は、プロジェクト内で表示されている最初の 10 個の run について、その設定やログ化されたメトリクスを 1 列ごとに表示します。

- 比較したい run を変更するには、画面左側の一覧で検索・フィルタ・グループ化・ソートができます。Run Comparer は自動で更新されます。
- 設定キーを検索またはフィルタしたい場合は、Run Comparer 上部の検索フィールドを使ってください。
- 違いだけを素早く確認したい場合や、同じ値を非表示にしたい場合は、パネル上部の **Diff only** を切り替えてください。
- 列の幅や行の高さを調整するには、パネル上部のフォーマットボタンを使用してください。
- 設定やメトリクスの値をコピーしたいときは、値の上にマウスカーソルを乗せ、コピーアイコンをクリックします。画面上で表示しきれないほど長い値でも、全体がコピーされます。

{{% alert %}}
デフォルトでは、Run Comparer は [`job_type`]({{< relref "/ref/python/sdk/functions/init.md" >}}) が異なる run を区別しません。つまり、プロジェクト内で比較可能でない run であっても比較できてしまう場合があります。例えば、training run と model evaluation run を比較することも可能です。training run には run のログ、ハイパーパラメーター、トレーニングの損失メトリクス、モデル本体などが含まれます。評価 run では、そのモデルを用いて新しいトレーニングデータ上でモデルのパフォーマンスを確認できます。

Runs Table 内で run のリストを検索・フィルタ・グループ化・ソートすると、Run Comparer は自動的に最初の 10 件を比較対象として更新します。`job_type` でフィルタやソートをすることで、似た run 同士の比較ができます。詳しくは[run のフィルタ方法]({{< relref "/guides/models/track/runs/filter-runs.md" >}})をご覧ください。
{{% /alert %}}