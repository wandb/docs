---
title: スムーズなラインプロット
description: ノイズの多いデータにおけるトレンドを見るために、線グラフでスムージングを使用します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B は 3 つのタイプの平滑化をサポートしています:

- [指数移動平均]({{< relref path="smoothing.md#exponential-moving-average-default" lang="ja" >}}) (デフォルト)
- [ガウス平滑化]({{< relref path="smoothing.md#gaussian-smoothing" lang="ja" >}})
- [移動平均]({{< relref path="smoothing.md#running-average" lang="ja" >}})
- [指数移動平均 - Tensorboard]({{< relref path="smoothing.md#exponential-moving-average-deprecated" lang="ja" >}}) (非推奨)

これらが [インタラクティブな W&B レポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)でどのように動作するかをご覧ください。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="" >}}

## 指数移動平均 (デフォルト)

指数平滑化は、時系列データを指数的に減衰させることで、過去のデータポイントの重みを滑らかにする手法です。範囲は 0 から 1 です。背景については [指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing) をご覧ください。時系列の初期値がゼロに偏らないようにするためのデバイアス項が追加されています。

EMA アルゴリズムは、線上の点の密度（x 軸範囲の単位当たりの `y` 値の数）を考慮に入れます。これにより、異なる特性を持つ複数の線を同時に表示する際に、一貫した平滑化が可能になります。

これが内部でどのように動作するかのサンプルコードです:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE は結果をチャートの x 軸範囲にスケーリングします
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

これがアプリ内でどのように見えるかはこちらをご覧ください [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="" >}}

## ガウス平滑化

ガウス平滑化（またはガウスカーネル平滑化）は、標準偏差が平滑化パラメータとして指定されるガウス分布に対応する重みを用いてポイントの加重平均を計算します。入力 x 値ごとに平滑化された値が計算されます。

ガウス平滑化は、TensorBoard の振る舞いと一致させる必要がない場合の標準的な選択肢です。指数移動平均とは異なり、ポイントは前後の値に基づいて平滑化されます。

これがアプリ内でどのように見えるかはこちらをご覧ください [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="" >}}

## 移動平均

移動平均は、与えられた x 値の前後のウィンドウ内のポイントの平均でそのポイントを置き換える平滑化アルゴリズムです。詳細は "Boxcar Filter" を参照してください [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)。移動平均のために選択されたパラメータは、Weights and Biases に移動平均で考慮するポイントの数を伝えます。

ポイントが x 軸上で不均一に配置されている場合は、ガウス平滑化を検討してください。

次の画像は、アプリ内での移動アプリの表示例を示しています [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

{{< img src="/images/app_ui/running_average.png" alt="" >}}

## 指数移動平均 (非推奨)

> TensorBoard EMA アルゴリズムは、同じチャート上で一貫したポイント密度を持たない複数の線を正確に平滑化することができないため、非推奨とされました。

指数移動平均は、TensorBoard の平滑化アルゴリズムと一致するように実装されています。範囲は 0 から 1 です。背景については [指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing) をご覧ください。時系列の初期値がゼロに偏らないようにするためのデバイアス項が追加されています。

これが内部でどのように動作するかのサンプルコードです:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

これがアプリ内でどのように見えるかはこちらをご覧ください [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/exponential_moving_average.png" alt="" >}}

## 実装の詳細

すべての平滑化アルゴリズムはサンプリングされたデータで実行されます。つまり、1500 ポイント以上をログに記録した場合、平滑化アルゴリズムはサーバーからポイントがダウンロードされた後に実行されます。平滑化アルゴリズムの目的は、データ内のパターンを迅速に見つけることです。多くのログを持つメトリクスに対して正確な平滑化された値が必要な場合は、API を介してメトリクスをダウンロードし、自分自身の平滑化メソッドを実行する方が良いかもしれません。

## 元のデータを非表示にする

デフォルトでは、オリジナルの非平滑化データを背景として薄い線で表示します。この表示をオフにするには、**Show Original** トグルをクリックしてください。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="" >}}