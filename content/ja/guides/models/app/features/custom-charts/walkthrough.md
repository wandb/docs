---
title: 'チュートリアル: カスタムチャートの使い方'
description: W&B UI のカスタムチャート機能を使ったチュートリアル
menu:
  default:
    identifier: ja-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

カスタムチャートを使うと、パネルに表示するデータや可視化を自由にコントロールできます。

## 1. W&B へのデータのログ

まず、スクリプトでデータをログします。トレーニング開始時に設定するハイパーパラメーターなどの単一ポイントには [wandb.Run.config]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) を、時間経過で複数のポイントやカスタム2次元配列のログには [wandb.Run.log()]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を使います。`wandb.Table()` でカスタムデータをテーブルとしてログできます。1つのキーにつき最大10,000データポイントまでのログを推奨しています。

```python
with wandb.init() as run: 

  # カスタムデータテーブルのログ
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

[クイック例のノートブック](https://bit.ly/custom-charts-colab) でデータテーブルのログを試し、次のステップでカスタムチャートを作成しましょう。結果のチャートは [ライブレポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc) で見ることができます。

## 2. クエリを作成

データがログできたら、プロジェクトページへ行き、**`+`** ボタンをクリックして新規パネルを追加し、**Custom Chart** を選んでください。[カスタムチャートのデモワークスペース](https://app.wandb.ai/demo-team/custom-charts) で手順を追うこともできます。

{{< img src="/images/app_ui/create_a_query.png" alt="ブランクのカスタムチャート" >}}

### クエリを追加

1. `summary` をクリックし、 `historyTable` を選んで run の履歴からデータを取得する新しいクエリを設定します。
2. `wandb.Table()` をログしたキーを入力します。上記コードスニペットでは `my_custom_table` です。[例のノートブック](https://bit.ly/custom-charts-colab) では、`pr_curve` と `roc_curve` がキーになっています。

### Vega フィールドの設定

クエリからこれらのカラムが読み込まれると、Vega フィールドのドロップダウンで利用可能になります。

{{< img src="/images/app_ui/set_vega_fields.png" alt="クエリ結果からカラムを選んで Vega フィールドに設定" >}}

* **x軸:** runSets_historyTable_r (recall)
* **y軸:** runSets_historyTable_p (precision)
* **color:** runSets_historyTable_c (クラスラベル)

## 3. チャートをカスタマイズ

かなりいい感じですが、散布図から折れ線グラフに切り替えたい場合、**Edit** をクリックして、Vega スペックを編集します。[カスタムチャートのデモワークスペース](https://app.wandb.ai/demo-team/custom-charts) で一緒に試すこともできます。

{{< img src="/images/general/custom-charts-1.png" alt="カスタムチャートの選択" >}}

Vega スペックのカスタマイズ例：

* グラフ、凡例、x軸、y軸にタイトルを追加（各フィールドで “title” を設定）
* “mark” の値を “point” から “line” に変更
* 未使用の “size” フィールドを削除

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="PRカーブ Vega スペック" >}}

この設定を今いるプロジェクト内で再利用したい場合は、ページ上部の **Save as** をクリックしてプリセットとして保存できます。下記のように、ROCカーブも一緒に表示できます。

{{< img src="/images/general/custom-charts-2.png" alt="PRカーブチャート" >}}

## ボーナス：合成ヒストグラム

ヒストグラムは数値分布を可視化し、大きなデータセットの理解に役立ちます。合成ヒストグラムでは複数の分布を同じビンで見比べることができ、異なるモデル同士や同じモデル内の異なるクラスのメトリクス比較ができます。例えば、運転シーンの物体検出用セマンティックセグメンテーションモデルなら、精度最適化と intersection over union (IoU) 最適化の効果や、モデルごとに車（データ中で大きく一般的な領域）と標識（より小さく稀な領域）検出を比較することができます。[デモ Colab](https://bit.ly/custom-charts-colab) では、10クラス中2クラスの生物への信頼度スコア比較が試せます。

{{< img src="/images/app_ui/composite_histograms.png" alt="合成ヒストグラム" >}}

自分だけのカスタム合成ヒストグラムパネルを作成するには：

1. Workspace または Report で新しい Custom Chart パネルを作成します（“Custom Chart” の可視化を追加）。右上の “Edit” ボタンをクリックして、任意の組み込みパネル種別から Vega spec を編集します。
2. 組み込みの Vega spec を[合成ヒストグラム用の MVP コード（Vega）](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)に置き換えます。メインタイトルや軸タイトル、入力範囲などを [Vega の構文](https://vega.github.io/) で直接編集できます（色の変更や、3つ目のヒストグラム追加もOKです）。
3. 右側のクエリ欄で wandb のログから正しいデータを取得するように設定します。`summaryTable` フィールドを追加し、対応する `tableKey` を `class_scores` に設定して、run でログされた `wandb.Table` を取得します。これにより、`red_bins` と `blue_bins` の2つのヒストグラムビンセットに、`class_scores` としてログされた `wandb.Table` のカラムをドロップダウンから割り当て可能になります。例では、赤いビンに `animal` クラスの予測スコア、青いビンに `plant` クラスのものを選びました。
4. Vega spec やクエリを何度でも調整でき、プレビューで満足のいくグラフになるまで変更できます。終わったら、画面上部の **Save as** を押してカスタムグラフに名前を付けて保存できます。その後 **Apply from panel library** をクリックすれば完成です。

たった1回、1000枚で1エポックだけトレーニングした簡単な実験の結果は以下の通りです。モデルは「ほとんどの画像は植物ではない」ととても確信し、「どれが動物か」についてはかなり曖昧でした。

{{< img src="/images/general/custom-charts-3.png" alt="チャート設定" >}}

{{< img src="/images/general/custom-charts-4.png" alt="チャート結果" >}}