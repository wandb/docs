---
title: 折れ線グラフの平滑化
description: 折れ線グラフでは、ノイズの多いデータの傾向を見るためにスムージングを使いましょう。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B では、いくつかの種類の平滑化をサポートしています:

- [時間重み付き指数移動平均 (TWEMA) の平滑化]({{< relref path="#time-weighted-exponential-moving-average-twema-smoothing-default" lang="ja" >}}) 
- [ガウシアン平滑化]({{< relref path="#gaussian-smoothing" lang="ja" >}})
- [移動平均の平滑化]({{< relref path="#running-average-smoothing" lang="ja" >}})
- [指数移動平均 (EMA) の平滑化]({{< relref path="#exponential-moving-average-ema-smoothing" lang="ja" >}})

これらの動作は [インタラクティブな W&B report](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) で確認できます。

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="さまざまな平滑化アルゴリズムのデモ" >}}

## 時間重み付き指数移動平均 (TWEMA) の平滑化（デフォルト）

時間重み付き指数移動平均 (TWEMA) の平滑化アルゴリズムは、過去の点の重みを指数的に減衰させることで時系列 データを平滑化する手法です。手法の詳細は [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing) を参照してください。範囲は 0 から 1 です。時系列の初期の値が 0 に偏らないよう、バイアス補正項が追加されています。

TWEMA アルゴリズムは、線上の点の密度（x 軸上の範囲の単位あたりの `y` 値の数）を考慮に入れます。これにより、特性の異なる複数の線を同時に表示する際でも、一貫した平滑化が可能になります。

内部でどのように動作するかのサンプル コードは次のとおりです:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE は結果をチャートの x 軸の範囲にスケールします
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

アプリ内での表示は [こちら](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="TWEMA 平滑化のデモ" >}}

## ガウシアン平滑化

ガウシアン平滑化（ガウシアン カーネル平滑化）は、標準偏差を平滑化パラメータとして指定したガウシアン分布に対応する重みで点の加重平均を計算します。平滑化された値は、前後の点に基づいて、入力された各 x 値に対して計算されます。

アプリ内での表示は [こちら](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="ガウシアン平滑化のデモ" >}}

## 移動平均の平滑化

移動平均は、与えられた x 値の前後のウィンドウ内の点の平均でその点を置き換える平滑化アルゴリズムです。詳細は Wikipedia の [“Boxcar Filter”](https://en.wikipedia.org/wiki/Moving_average) を参照してください。移動平均で選択したパラメータは、Weights and Biases が移動平均で考慮する点の数を指定します。

x 軸上で点の間隔が不均一な場合は、代わりにガウシアン平滑化の使用を検討してください。

アプリ内での表示は [こちら](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

{{< img src="/images/app_ui/running_average.png" alt="移動平均平滑化のデモ" >}}

## 指数移動平均 (EMA) の平滑化

指数移動平均 (EMA) の平滑化アルゴリズムは、指数ウィンドウ関数を用いて時系列 データを平滑化する経験則的な手法です。手法の詳細は [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing) を参照してください。範囲は 0 から 1 です。時系列の初期の値が 0 に偏らないように、バイアス補正項が追加されています。

多くの場合、EMA の平滑化は、先にバケット化してから平滑化するのではなく、履歴全体を通して適用されます。こちらのほうが、より正確な平滑化になることが多いです。

ただし、次の場合は EMA の平滑化は先にバケット化を行った後に適用されます:
- Sampling
- Grouping
- Expressions
- Non-monotonic x-axes
- Time-based x-axes

内部でどのように動作するかのサンプル コードは次のとおりです:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

アプリ内での表示は [こちら](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/exponential_moving_average.png" alt="EMA 平滑化のデモ" >}}

## 元のデータを非表示にする

デフォルトでは、平滑化していない元のデータが、背景の薄い線としてプロットに表示されます。これを無効にするには、**Show Original** をクリックします。

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="元のデータの表示をオン/オフする" >}}