---
title: Smooth line plots
description: 折れ線グラフでは、スムージングを使ってノイズの多いデータ のトレンドを確認します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B は3種類の平滑化をサポートしています。

- [指数移動平均]({{< relref path="smoothing.md#exponential-moving-average-default" lang="ja" >}}) (デフォルト)
- [ガウシアン平滑化]({{< relref path="smoothing.md#gaussian-smoothing" lang="ja" >}})
- [移動平均]({{< relref path="smoothing.md#running-average" lang="ja" >}})
- [指数移動平均 - Tensorboard]({{< relref path="smoothing.md#exponential-moving-average-deprecated" lang="ja" >}}) (非推奨)

これらの機能を [インタラクティブな W&B レポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) でライブで確認できます。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="" >}}

## 指数移動平均 (デフォルト)

指数平滑化は、過去の点の重みを指数関数的に減衰させることで、 時系列 データを平滑化する手法です。範囲は0から1です。背景については、[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)を参照してください。時系列の初期の値がゼロに偏らないように、バイアス除去項が追加されています。

EMAアルゴリズムは、線上の点の密度 (x軸の範囲の単位あたりの `y` 値の数) を考慮します。これにより、異なる特性を持つ複数の線を同時に表示する際に、一貫した平滑化が可能になります。

以下は、この仕組みの内部動作を示すサンプル コードです。

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALEは、結果をチャートのx軸範囲にスケールします
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

これは [アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) では次のようになります。

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="" >}}

## ガウシアン平滑化

ガウシアン平滑化 (またはガウシアン カーネル平滑化) は、点の加重平均を計算します。ここで、重みは平滑化 パラメータ として指定された標準偏差を持つガウス分布に対応します。を参照してください。平滑化された 値 は、すべての入力x 値 に対して計算されます。

TensorBoard の 振る舞い と一致させることを気にしない場合は、ガウシアン平滑化は平滑化に適した標準的な選択肢です。指数移動平均とは異なり、値の前後に発生する点に基づいて点が平滑化されます。

これは [アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing) では次のようになります。

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="" >}}

## 移動平均

移動平均は、指定されたx 値 の前後のウィンドウ内の点の平均で点を置き換える平滑化アルゴリズムです。[https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average) の「Boxcar Filter」を参照してください。移動平均に選択された パラメータ は、Weights and Biases に移動平均で考慮する点の数を伝えます。

点 がx軸上で不均等に配置されている場合は、ガウシアン平滑化の使用を検討してください。

次の画像は、実行中のアプリが [アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average) でどのように見えるかを示しています。

{{< img src="/images/app_ui/running_average.png" alt="" >}}

## 指数移動平均 (非推奨)

> TensorBoard EMAアルゴリズムは、一貫した点密度 (x軸の単位あたりにプロットされる点の数) を持たない同じチャート上の複数の線を正確に平滑化できないため、非推奨になりました。

指数移動平均は、TensorBoard の平滑化アルゴリズムと一致するように実装されています。範囲は0から1です。背景については、[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)を参照してください。時系列の初期の値がゼロに偏らないように、バイアス除去項が追加されています。

以下は、この仕組みの内部動作を示すサンプル コードです。

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

これは [アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) では次のようになります。

{{< img src="/images/app_ui/exponential_moving_average.png" alt="" >}}

## 実装の詳細

すべての平滑化アルゴリズムはサンプリングされた データ で実行されます。つまり、1500を超える点を ログ に記録すると、平滑化アルゴリズムは サーバー から点がダウンロードされた _後_ に実行されます。平滑化アルゴリズムの目的は、 データ 内のパターンをすばやく見つけるのに役立つことです。多数の ログ に記録された点 を持つ メトリクス で正確な平滑化された 値 が必要な場合は、API を介して メトリクス をダウンロードし、独自の平滑化 メソッド を実行する方が良い場合があります。

## 元のデータを非表示にする

デフォルトでは、元の平滑化されていない データ が背景に薄い線として表示されます。これをオフにするには、[元のデータを表示] トグルをクリックします。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="" >}}
