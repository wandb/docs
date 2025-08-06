---
title: 滑らかな折れ線グラフ
description: 折れ線グラフでは、スムージングを使ってノイズの多いデータの傾向を確認できます。
menu:
  default:
    identifier: smoothing
    parent: line-plot
weight: 30
---

W&B では、以下の複数のスムージング手法をサポートしています。

- [時間重み付き指数移動平均（TWEMA）スムージング]({{< relref "#time-weighted-exponential-moving-average-twema-smoothing-default" >}})
- [ガウススムージング]({{< relref "#gaussian-smoothing" >}})
- [ランニングアベレージ]({{< relref "#running-average-smoothing" >}})
- [指数移動平均（EMA）スムージング]({{< relref "#exponential-moving-average-ema-smoothing" >}})

これらの実際の動作は[インタラクティブな W&B レポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)で見ることができます。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="様々なスムージングアルゴリズムのデモ" >}}

## 時間重み付き指数移動平均（TWEMA）スムージング（デフォルト）

時間重み付き指数移動平均（TWEMA）スムージングアルゴリズムは、時系列データを平滑化するために、過去の点の重みを指数関数的に減少させる手法です。技術的な詳細は [指数平滑法](https://www.wikiwand.com/en/Exponential_smoothing) を参照してください。範囲は 0 から 1 です。時系列の初期値がゼロにバイアスされないよう補正項が加えられています。

TWEMA フィルターは、線上の点の密度（x 軸単位あたりの `y` 値の数）を考慮します。このため、異なる特徴を持つ複数線の同時表示でも一貫したスムージングが可能です。

以下は内部で動作する仕組みのサンプルコードです：

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE はチャートの x 軸の範囲に合わせてスケールします
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

このアルゴリズムの実際の表示例は[アプリ上で確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="TWEMAスムージングのデモ" >}}

## ガウススムージング

ガウススムージング（またはガウスカーネル平滑化）は、各点に対してガウス分布に基づいた重み付き平均を計算します。スムージングパラメータで標準偏差を指定し、その範囲内にある前後両方の点を使って各入力 x の値ごとに平滑化された値を計算します。

このアルゴリズムの実際の表示例は[アプリ上で確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="ガウススムージングのデモ" >}}

## ランニングアベレージ・スムージング

ランニングアベレージは、指定された x 値の前後にウィンドウを取り、その範囲内の点の平均で各点を置き換えるスムージングアルゴリズムです。詳細は ["Boxcar Filter" の Wikipedia](https://en.wikipedia.org/wiki/Moving_average) をご覧ください。指定パラメータは、Weights and Biases において移動平均に含める点数を表します。

x 軸上で点の間隔が不均一な場合は、代わりにガウススムージングをご利用ください。

このアルゴリズムの実際の表示例は[アプリ上で確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

{{< img src="/images/app_ui/running_average.png" alt="ランニングアベレージ・スムージングのデモ" >}}

## 指数移動平均（EMA）スムージング

指数移動平均（EMA）スムージングアルゴリズムは、時系列データを指数関数的ウィンドウ関数で平滑化する一般的なテクニックです。技術的な詳細は [指数平滑法](https://www.wikiwand.com/en/Exponential_smoothing) をご覧ください。範囲は 0 から 1 です。時系列の初期値がゼロにバイアスされないよう補正項が加えられています。

多くの場合、EMA スムージングはまず全履歴に適用され、その後に区切る（バケットを作る）ことなくスムージングします。これにより、スムージングの精度が向上することがあります。

以下のような場合は、先にバケット化を行った後に EMA スムージングが適用されます：
- サンプリング
- グルーピング
- 数式（Expressions）
- x軸が単調増加しない場合
- 時間ベースの x軸

このアルゴリズムの内部動作を示すサンプルコードはこちらです：

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

このアルゴリズムの実際の表示例は[アプリ上で確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/exponential_moving_average.png" alt="EMAスムージングのデモ" >}}

## 元のデータを非表示にする

デフォルトでは、元の（平滑化されていない）データがプロットの背景に淡く表示されます。**Show Original** をクリックすると、この表示をオフにできます。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="元のデータを表示・非表示に切り替え" >}}