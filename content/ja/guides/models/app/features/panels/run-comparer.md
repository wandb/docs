---
title: run のメトリクスを比較する
description: 複数の run 間でメトリクスを比較する
menu:
  default:
    identifier: ja-guides-models-app-features-panels-run-comparer
    parent: panels
weight: 70
---

Run Comparer を使って、プロジェクト内の複数の run の違いや共通点を比較できます。

## Run Comparer パネルを追加する

1. ページ右上の **Add panels** ボタンを選択します。
1. **Evaluation** セクションから **Run comparer** を選択します。

## Run Comparer の使い方
Run Comparer では、プロジェクト内で表示されている最初の10件の run について、設定や記録されたメトリクスを run ごとに1列として表示します。

- 比較する run を変更したい場合は、左側の run 一覧で検索、フィルター、グループ化、並び替えができます。Run Comparer の内容は自動で更新されます。
- 設定のキーでフィルターや検索をしたい場合は、Run Comparer 上部の検索フィールドを使ってください。
- 違いだけを素早く確認したいときや、同じ値を非表示にしたい場合は、パネル上部の **Diff only** を切り替えてください。
- 列幅や行の高さを調整するには、パネル上部のフォーマット用ボタンを使用します。
- 設定やメトリクスの値をコピーしたいときは、値の上にマウスを重ねてからコピーのボタンをクリックします。画面に収まらない長い値でも全体がコピーされます。

{{% alert %}}
デフォルトでは、Run Comparer は [`job_type`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の値が異なる run を区別しません。つまり、プロジェクト内で本来比較できない run 同士も比較できてしまう場合があります。たとえば、training run と model evaluation run を比較することも可能です。training run には run のログやハイパーパラメーター、トレーニングの損失メトリクス、モデルそのものなどが含まれます。evaluation run では、そのモデルを使って新しいトレーニングデータに対するモデルの性能をチェックできます。

Runs Table で run の検索、フィルター、グループ化、並び替えを行うと、Run Comparer も自動で最初の10件の run を比較するように更新されます。たとえば `job_type` で一覧をフィルター・並び替えすることで、似た run 同士を簡単に比較できます。詳しくは [run のフィルタリング方法]({{< relref path="/guides/models/track/runs/filter-runs.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}