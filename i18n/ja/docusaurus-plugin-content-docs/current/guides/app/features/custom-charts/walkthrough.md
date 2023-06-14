---
description: Tutorial of using the custom charts feature in the Weights & Biases UI
displayed_sidebar: ja
---

# カスタムチャートのウォークスルー

Weights & Biasesの組み込みチャートを超えて、新しい**カスタムチャート**機能を使って、パネルにロードするデータの詳細やデータの可視化方法を制御します。

**概要**

1. W&Bにデータを記録
2. クエリを作成
3. チャートをカスタマイズ

## 1. W&Bにデータを記録

まず、スクリプト内でデータを記録します。[wandb.config](../../../../guides/track/config.md)をトレーニングの最初に設定された単一のポイント、例えばハイパーパラメータのために使用します。[wandb.log()](../../../../guides/track/log/intro.md)を時間をかけて複数のポイントのために使用し、wandb.Table()でカスタム2D配列を記録します。記録されたキーごとに最大10,000個のデータポイントを記録することをお勧めします。

```python
# カスタムデータテーブルの記録
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log({"custom_data_table": wandb.Table(data=my_custom_data,
                                columns = ["x", "y", "z"])})
```

データテーブルを記録するための[クイックな例ノートブック](https://bit.ly/custom-charts-colab)を試してみてください。次のステップではカスタムチャートを設定します。[ライブレポート](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)で結果のチャートを確認してください。

## 2. クエリを作成

可視化するデータを記録したら、プロジェクトページに移動し、**`+`**ボタンをクリックして新しいパネルを追加し、**カスタムチャート** を選択します。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts)で一緒にフォローしてください。

![新しい、空白のカスタムチャートが設定される準備が整っています](/images/app_ui/create_a_query.png)
### クエリを追加

1. `summary`をクリックし、`historyTable`を選択して、run履歴からデータを取得する新しいクエリを設定します。
2. **wandb.Table()**をログしたキーを入力します。上記のコードスニペットでは、`my_custom_table`でした。[サンプルノートブック](https://bit.ly/custom-charts-colab)では、キーは`pr_curve`と`roc_curve`です。

### Vegaフィールドの設定

クエリがこれらの列をロードしているので、Vegaフィールドのドロップダウンメニューで選択できるオプションとして利用できます。

![クエリの結果から列を引いてVegaフィールドを設定する](/images/app_ui/set_vega_fields.png)

* **x軸:** runSets\_historyTable\_r (recall)
* **y軸:** runSets\_historyTable\_p (precision)
* **色:** runSets\_historyTable\_c (class label)

## 3. チャートのカスタマイズ

これでかなり良い感じになりましたが、散布図から折れ線グラフに切り替えたいと思います。**編集**をクリックして、このビルトインチャートのVega仕様を変更します。[このワークスペース](https://app.wandb.ai/demo-team/custom-charts)で手順に沿って進めてください。

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1597442115525\_Screen+Shot+2020-08-14+at+2.52.24+PM.png)

Vegaの仕様を更新して可視化をカスタマイズしました。

* プロット、凡例、x軸、y軸のタイトルを追加（各フィールドに"title"を設定）
* “mark”の値を“point”から“line”に変更
* 未使用の“size”フィールドを削除

![](/images/app_ui/customize_vega_spec_for_pr_curve.png)

このプロジェクト内で他の場所で使用できるプリセットとして保存するには、ページの上部にある**Save as**をクリックします。これが結果の見た目です。ROC曲線と一緒に表示されます。

![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1597442868347\_Screen+Shot+2020-08-14+at+3.07.30+PM.png)
## ボーナス: 複合ヒストグラム



ヒストグラムは数値分布を視覚化し、より大きなデータセットを理解するのに役立ちます。複合ヒストグラムは、同じビンに対する複数の分布を示し、異なるモデルや、モデル内の異なるクラス間で2つ以上の指標を比較することができます。運転シーン内のオブジェクトを検出するセマンティックセグメンテーションモデルでは、精度と交差オーバーユニオン（IOU）の最適化の効果を比較することができますし、異なるモデルが自動車（データ内の大きくて一般的な地域）と交通標識（はるかに小さくて一般的でない地域）をどの程度検出するかを知りたいかもしれません。 [デモColab](https://bit.ly/custom-charts-colab)では、生き物の10のクラスのうち2つの信頼スコアを比較することができます。



![](/images/app_ui/composite_histograms.png)



カスタム複合ヒストグラムパネルの独自バージョンを作成するには、以下の手順に従ってください。



1. ワークスペースまたはレポートに新しいカスタムチャートパネルを作成します（「カスタムチャート」の可視化を追加）。右上の[編集]ボタンを押して、任意のビルトインパネルタイプからVega仕様を変更できます。

2. そのビルトインのVega仕様を、[複合ヒストグラム用のVegaのMVPコード](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)に置き換えます。このVega仕様で、[Vega構文](https://vega.github.io/)を使用して、メインタイトル、軸タイトル、入力ドメイン、その他の詳細を直接編集できます。（色を変更したり、3つ目のヒストグラムを追加したりすることもできます :)）

3. 右側のクエリを修正して、wandbのログから正しいデータをロードします。「summaryTable」フィールドを追加し、「tableKey」を「class\_scores」に設定して、runによってログされたwandb.Tableを取得します。これにより、「red\_bins」と「blue\_bins」という2つのヒストグラムビンセットを、「class\_scores」としてログされたwandb.Tableの列を使ってドロップダウンメニューで設定できるようになります。私の例では、「animal」クラスの予測スコアを赤いビンに、「plant」を青いビンに選択しました。

4. Vega仕様とクエリを変更し続けて、プレビューレンダリングで表示されるプロットに満足するまで続けます。完了したら、上部の「名前を付けて保存」をクリックして、カスタムプロットに名前を付けて再利用できるようにします。次に、「パネルライブラリから適用」をクリックしてプロットを完成させます。

私の結果は、非常に短い実験から得られたものです。たった1000例のうち１エポックだけ学習させたモデルは、ほとんどの画像が植物でないことに非常に確信を持っており、どの画像が動物かどうかについては非常に不確かです。



![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1598376315319\_Screen+Shot+2020-08-25+at+10.24.49+AM.png)



![](https://paper-attachments.dropbox.com/s\_5FCA7E5A968820ADD0CD5402B4B0F71ED90882B3AC586103C1A96BF845A0EAC7\_1598376160845\_Screen+Shot+2020-08-25+at+10.08.11+AM.png)