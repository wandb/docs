---
description: In line plots, use smoothing to see trends in noisy data.
displayed_sidebar: ja
---

# スムージング

Weights＆Biasesの線グラフでは、以下の3種類のスムージングをサポートしています:

* [指数移動平均](smoothing.md#exponential-moving-average-default) (デフォルト)
* [ガウススムージング](smoothing.md#gaussian-smoothing)
* [移動平均](smoothing.md#running-average)

これらについては、[インタラクティブなW&Bレポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)で実際に確認できます。

![](/images/app_ui/beamer_smoothing.gif)

## 指数移動平均 (デフォルト)

指数移動平均は、TensorBoardのスムージングアルゴリズムに合わせて実装されています。範囲は0から1です。「[指数平滑化](https://www.wikiwand.com/en/Exponential\_smoothing)」を参照してください。最初の時系列データの値がゼロにバイアスされないように、デバイアス項が追加されています。

これが内部的にどのように機能しているかのサンプルコードは次のとおりです。

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```
このように見えます[アプリで](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)：

![](/images/app_ui/exponential_moving_average.png)

## ガウシアンスムージング

ガウシアンスムージング（またはガウシアンカーネルスムージング）は、ポイントの重み付け平均を計算し、重みはスムージングパラメータとして指定された標準偏差を持つガウシアン分布に対応します。参照してください。スムージングされた値は、すべての入力x値に対して計算されます。

ガウシアンスムージングは、テンソルボードの動作と一致することに関心がない場合、スムージングに適した標準的な選択です。指数移動平均とは異なり、ポイントはその前後の両方の値によってスムージングされるでしょう。

これが[アプリで](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing)どのように見えるかです：

![](/images/app_ui/gaussian_smoothing.png)

## 移動平均

移動平均は、単純なスムージングアルゴリズムであり、与えられたx値の前後のウィンドウ内のポイントの平均でポイントを置き換えます。[https://en.wikipedia.org/wiki/Moving\_average](https://en.wikipedia.org/wiki/Moving\_average) の "Boxcar Filter" を参照してください。移動平均を考慮するポイント数をWeights and Biasesに伝えるために選択されたパラメータです。

移動平均は、単純で複製しやすいスムージングアルゴリズムです。ポイントがx軸上で不均一に配置されている場合、ガウシアンスムージングがより適切な選択となります。

これが[アプリで](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average)どのように見えるかです：

![](/images/app_ui/running_average.png)

## 実装の詳細

すべてのスムージングアルゴリズムは、サンプルデータで実行されるため、3000ポイント以上ログする場合、スムージングアルゴリズムはサーバーからのポイントのダウンロードが完了した後に実行されます。スムージングアルゴリズムの目的は、データのパターンを迅速に見つけることを支援することです。ログポイント数の多いメトリクス上で正確なスムーズ化された値が必要な場合は、APIを介してメトリクスをダウンロードし、独自のスムージング方法を実行する方が良いかもしれません。

## オリジナルデータを非表示にする
デフォルトでは、元の滑らかでないデータを背景に薄い線で表示しています。この表示をオフにするには、**Show Original** トグルをクリックしてください。



![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)