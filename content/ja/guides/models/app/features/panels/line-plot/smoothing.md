---
title: Smooth line plots
description: 折れ線グラフでは、平滑化を使用してノイズの多いデータ の傾向を確認します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B は、3種類のスムージングをサポートしています。

- [指数移動平均]({{< relref path="smoothing.md#exponential-moving-average-default" lang="ja" >}}) （デフォルト）
- [ガウス スムージング]({{< relref path="smoothing.md#gaussian-smoothing" lang="ja" >}})
- [移動平均]({{< relref path="smoothing.md#running-average" lang="ja" >}})
- [指数移動平均 - Tensorboard]({{< relref path="smoothing.md#exponential-moving-average-deprecated" lang="ja" >}}) (非推奨)

[インタラクティブな W&B report](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) でこれらのライブをご覧ください。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="" >}}

## 指数移動平均（デフォルト）

指数平滑化は、過去の点の重みを指数関数的に減衰させることによって、時系列データを平滑化する手法です。範囲は0〜1です。背景については、[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)をご覧ください。時系列の初期値がゼロに偏らないように、バイアス除去項が追加されています。

EMA アルゴリズムは、線上の点の密度（x軸の範囲の単位あたりの `y` 値の数）を考慮します。これにより、異なる特性を持つ複数の線を同時に表示する場合でも、一貫したスムージングが可能になります。

以下は、この仕組みの内部動作を示すサンプル コードです。

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE scales the result to the chart's x-axis range
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

[アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) での表示は次のとおりです。

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="" >}}

## ガウス スムージング

ガウス スムージング（またはガウス カーネル スムージング）は、点の加重平均を計算します。重みは、平滑化 パラメータとして指定された標準偏差を持つガウス分布に対応します。 を参照してください。平滑化された値は、すべての入力 x 値に対して計算されます。

TensorBoard の振る舞いとの一致を気にしない場合は、ガウス スムージングはスムージングの標準的な選択肢として適しています。指数移動平均とは異なり、ポイントは値の前後に発生するポイントに基づいて平滑化されます。

[アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing) での表示は次のとおりです。

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="" >}}

## 移動平均

移動平均は、指定された x 値の前後のウィンドウ内の点の平均で点を置き換える平滑化アルゴリズムです。[https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average) の「Boxcar Filter」を参照してください。移動平均に選択されたパラメータは、移動平均で考慮する点の数を Weights and Biases に伝えます。

ポイントが x 軸上で不均等に配置されている場合は、ガウス スムージングの使用を検討してください。

次の画像は、実行中のアプリが [アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average) でどのように見えるかを示しています。

{{< img src="/images/app_ui/running_average.png" alt="" >}}

## 指数移動平均（非推奨）

> TensorBoard EMA アルゴリズムは、一貫したポイント密度（x 軸の単位あたりにプロットされるポイントの数）を持たない同じグラフ上の複数の線を正確に平滑化できないため、非推奨になりました。

指数移動平均は、TensorBoard のスムージング アルゴリズムと一致するように実装されています。範囲は0〜1です。背景については、[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)をご覧ください。時系列の初期値がゼロに偏らないように、バイアス除去項が追加されています。

以下は、この仕組みの内部動作を示すサンプル コードです。

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

[アプリ内](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) での表示は次のとおりです。

{{< img src="/images/app_ui/exponential_moving_average.png" alt="" >}}

## 実装の詳細

すべてのスムージング アルゴリズムはサンプル データで実行されます。つまり、1500 ポイントを超えるポイントをログに記録すると、スムージング アルゴリズムは、ポイントがサーバーからダウンロードされた _後_ に実行されます。スムージング アルゴリズムの目的は、データ内のパターンをすばやく見つけるのに役立つことです。多数のログに記録されたポイントを持つメトリクスの正確な平滑化された値が必要な場合は、API を介してメトリクスをダウンロードし、独自のスムージング method を実行する方が良い場合があります。

## 元のデータを非表示にする

デフォルトでは、平滑化されていない元のデータが背景に薄い線として表示されます。これをオフにするには、[元のデータを表示] トグルをクリックします。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="" >}}
