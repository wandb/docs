---
description: W&B UI でのカスタムチャート機能のチュートリアル
displayed_sidebar: default
---


# Custom Charts Walkthrough

W&B の組み込みチャートを超えて、新しい **Custom Charts** 機能を使って、パネルに読み込むデータの詳細とデータの可視化方法を制御します。

**概要**

1. W&B にデータをログ
2. クエリの作成
3. チャートのカスタマイズ

## 1. W&B にデータをログ

まず、スクリプトにデータをログします。[wandb.config](../../../../guides/track/config.md) を使用して、ハイパーパラメーターのようなトレーニング開始時に設定される単一のポイントを設定します。時間の経過とともに複数のポイントをログするために [wandb.log()](../../../../guides/track/log/intro.md) を使用し、wandb.Table() でカスタム2D配列をログします。ログされたキーごとに最大10,000データポイントのログを推奨します。

```python
# カスタムデータ表をログする
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

データテーブルをログするための [クイック例ノートブックを試してください](https://bit.ly/custom-charts-colab)。次のステップではカスタムチャートを設定します。[ライブレポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc) で結果のチャートを見ることができます。

## 2. クエリの作成

データを可視化するためにログしたら、プロジェクトページに移動して **`+`** ボタンをクリックして新しいパネルを追加し、**Custom Chart** を選択します。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts) に従って進めることができます。

![構成準備完了の新しい、空のカスタムチャート](/images/app_ui/create_a_query.png)

### クエリを追加

1. `summary` をクリックし、`historyTable` を選択して、run の履歴からデータを引き出す新しいクエリを設定します。
2. ログされた **wandb.Table()** のキーを入力します。上記のコードスニペットでは `my_custom_table` です。[例のノートブック](https://bit.ly/custom-charts-colab) では、キーは `pr_curve` と `roc_curve` です。

### Vega フィールドの設定

このクエリがこれらの列を読み込むと、それらは Vega フィールドのドロップダウンメニューで選択可能なオプションとして利用できるようになります。

![クエリ結果から列を引き出して Vega フィールドを設定する](/images/app_ui/set_vega_fields.png)

* **x軸:** runSets\_historyTable\_r (再現率)
* **y軸:** runSets\_historyTable\_p (精度)
* **色:** runSets\_historyTable\_c (クラスラベル)

## 3. チャートのカスタマイズ

これでかなり良さそうですが、散布図から折れ線グラフに変更したいです。**Edit** をクリックして、この組み込みチャートの Vega スペックを変更します。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts) に従って進めてください。

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1597442115525\_Screen+Shot+2020-08-14+at+2.52.24+PM.png)

Vega スペックを更新して可視化をカスタマイズしました。

* プロット、凡例、x軸、y軸のタイトルを追加（各フィールドの "title" を設定）
* “mark” の値を “point” から “line” に変更
* 使われていない “size” フィールドを削除

![](/images/app_ui/customize_vega_spec_for_pr_curve.png)

これをこのプロジェクトの他の場所で使えるプリセットとして保存するには、ページの上部にある **Save as** をクリックします。以下が結果の例で、ROC 曲線も含まれています。

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1597442868347\_Screen+Shot+2020-08-14+at+3.07.30+PM.png)

## ボーナス: 合成ヒストグラム

ヒストグラムは数値分布を可視化し、より大きなデータセットを理解するのに役立ちます。合成ヒストグラムは、同じビンに複数の分布を表示し、異なるモデルまたはモデル内の異なるクラス間で二つ以上のメトリクスを比較することができます。セマンティックセグメンテーションモデルで自動車の走行シーンのオブジェクトを検出する場合、精度と Intersection over union (IoU) の最適化の有効性を比較することができます。また、異なるモデルが車 (データ中の大きく一般的な領域) と交通標識 (非常に小さく一般的でない領域) をどれだけうまく検出するかを知りたい場合もあります。[デモの Colab](https://bit.ly/custom-charts-colab) では、十分類のうちの二つのカテゴリの信頼スコアを比較できます。

![](/images/app_ui/composite_histograms.png)

カスタム合成ヒストグラムパネルの自分バージョンを作成するには：

1. Workspace または Report に新しい Custom Chart パネルを作成します（「Custom Chart」可視化を追加することで）。右上の「Edit」ボタンを押して、任意の組み込みパネルタイプから Vega スペックを変更します。
2. 組み込みの Vega スペックを [Vega の合成ヒストグラム用 MVP コード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21) に置き換えます。[Vega 構文](https://vega.github.io/) を使用して、この Vega スペック内のメインタイトル、軸タイトル、入力ドメイン、その他の詳細を直接修正できます（色を変えることや、さらなるヒストグラムの追加などが可能です）。
3. 右側のクエリを修正して、wandb ログから正しいデータを読み込みます。「summaryTable」フィールドを追加し、対応する「tableKey」を「class\_scores」に設定して、run によってログされた wandb.Table を取得します。これにより、ドロップダウンメニューを介して「class\_scores」としてログされた wandb.Table の列を使って、二つのヒストグラムビンセット（「red\_bins」と「blue\_bins」）を設定できます。私の例では、赤いビンに「animal」クラスの予測スコアを、青いビンに「plant」を選択しました。
4. プレビューで見えるプロットに満足するまで、Vega スペックとクエリを継続的に変更できます。完了したら、上部の「Save as」をクリックしてカスタムプロットに名前を付け、再利用できるようにします。その後、「Apply from panel library」をクリックしてプロットを完成させます。

以下は、非常に短い実験から得られた私の結果の例です：1000例で 1 エポックだけトレーニングした結果、ほとんどの画像が植物ではないと非常に自信を持ち、どの画像が動物であるかについては非常に不確かです。

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1598376315319\_Screen+Shot+2020-08-25+at+10.24.49+AM.png)

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1598376160845\_Screen+Shot+2020-08-25+at+10.08.11+AM.png)