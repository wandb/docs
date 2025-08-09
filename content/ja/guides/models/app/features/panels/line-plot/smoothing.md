---
title: スムーズなラインプロット
description: 折れ線グラフでは、スムージングを使ってノイズの多いデータの傾向を確認できます。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B では、以下の種類のスムージングをサポートしています。

- [時系列重み付き指数移動平均（TWEMA）スムージング]({{< relref path="#time-weighted-exponential-moving-average-twema-smoothing-default" lang="ja" >}})
- [ガウススムージング]({{< relref path="#gaussian-smoothing" lang="ja" >}})
- [ランニングアベレージ]({{< relref path="#running-average-smoothing" lang="ja" >}})
- [指数移動平均（EMA）スムージング]({{< relref path="#exponential-moving-average-ema-smoothing" lang="ja" >}})

これらのスムージング方法は [インタラクティブな W&B レポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) でもご覧いただけます。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="さまざまなスムージングアルゴリズムのデモ" >}}

## 時系列重み付き指数移動平均（TWEMA）スムージング（デフォルト）

時系列重み付き指数移動平均（TWEMA）スムージングアルゴリズムは、時系列データの過去の値の重みを指数的に減衰させることでデータを平滑化する手法です。技術詳細については [指数平滑法](https://www.wikiwand.com/en/Exponential_smoothing) をご覧ください。範囲は 0 から 1 です。時系列の初期値が 0 に偏らないように、デバイアス項が追加されています。

TWEMA アルゴリズムは、線上の点の密度（x軸の範囲あたりの `y` の値の数）を考慮します。これにより異なる特徴を持つ複数の線を同時に表示する際にも、一貫したスムージングが行えます。

この仕組みを裏でどう実装しているかのサンプルコードはこちらです：

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE は結果をチャートの x軸の範囲にスケーリングする定数
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

この動作例は [アプリ内でも確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)：

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="TWEMA スムージングのデモ" >}}

## ガウススムージング

ガウススムージング（ガウスカーネルスムージング）は、各点にガウス分布（平滑化パラメータとして与えた標準偏差）の重みをかけて加重平均を計算します。スムージングされた値は、前後双方の点に基づいた各入力 x 値で算出されます。

この動作例は [アプリ内でも確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing)：

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="ガウススムージングのデモ" >}}

## ランニングアベレージスムージング

ランニングアベレージは、指定された x の前後のウィンドウにある値の平均値で各点を置き換える平滑化アルゴリズムです。詳細は ["Boxcar Filter"（Wikipedia）](https://en.wikipedia.org/wiki/Moving_average) をご覧ください。ランニングアベレージのパラメータには、Weights & Biases に渡す移動平均で考慮する点数を指定します。

もし x軸上の点が不均一に並んでいる場合は、代わりにガウススムージングの利用をおすすめします。

この動作例は [アプリ内でも確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average)：

{{< img src="/images/app_ui/running_average.png" alt="ランニングアベレージスムージングのデモ" >}}

## 指数移動平均（EMA）スムージング

指数移動平均（EMA）スムージングアルゴリズムは、指数関数型ウィンドウ関数を用いて時系列データを平滑化する経験則的な手法です。技術詳細については [指数平滑法](https://www.wikiwand.com/en/Exponential_smoothing) をご覧ください。範囲は 0 から 1 です。時系列の最初の値が 0 に偏らないよう、デバイアス項が追加されています。

多くの場合、EMA スムージングはバケット化せずに全履歴に直接適用されることが多く、その方が高い精度で平滑化できます。

以下のようなケースでは、EMA スムージングは先にバケット化してから適用されます：
- サンプリング
- グルーピング
- 式（expressions）の利用
- 非単調な x軸
- 時間ベースの x軸

この仕組みを裏でどう実装しているかのサンプルコードはこちらです：

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

この動作例は [アプリ内でも確認できます](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)：

{{< img src="/images/app_ui/exponential_moving_average.png" alt="EMA スムージングのデモ" >}}

## 元データの表示/非表示

デフォルトで、元の未スムージングのデータはグラフの背景に薄く表示されます。**Show Original** をクリックするとこの表示を切り替えることができます。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="元データの表示/非表示切り替え" >}}