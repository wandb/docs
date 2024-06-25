---
description: '折れ線グラフでは、データのノイズを平滑化して傾向を確認します。

  '
displayed_sidebar: default
---


# 平滑化

W&Bの折れ線グラフでは、以下の3種類の平滑化をサポートしています。

- [指数移動平均](smoothing.md#exponential-moving-average-default) (デフォルト)
- [ガウス平滑化](smoothing.md#gaussian-smoothing)
- [移動平均](smoothing.md#running-average)
- [指数移動平均 - Tensorboard](smoothing.md#exponential-moving-average-tensorboard) (非推奨)

これらの平滑化を実際に確認するには、[インタラクティブなW&Bレポート](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)をご覧ください。

![](/images/app_ui/beamer_smoothing.gif)

## 指数移動平均 (デフォルト)

指数平滑化は、時系列データを指数的に古いデータの重みを減衰させることで平滑化する手法です。範囲は0から1までです。背景については[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)を参照してください。時系列の初期値がゼロに偏らないように、デバイアス項が追加されています。

EMAアルゴリズムは、折れ線の点の密度（つまり、x軸の範囲の単位あたりの`y`値の数）を考慮に入れます。これにより、異なる特性を持つ複数の線を同時に表示する際に一貫した平滑化が可能になります。

以下は、このアルゴリズムがどのように動作するかについてのサンプルコードです。

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE は結果をチャートのx軸範囲にスケールします
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

アプリでの表示例はこちらです。[in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/weighted_exponential_moving_average.png)

## ガウス平滑化

ガウス平滑化（またはガウスカーネル平滑化）は、ガウス分布に対応する重みで点の加重平均を計算します。平滑化パラメータとして標準偏差が指定されます。平滑化された値は、すべての入力x値に対して計算されます。

ガウス平滑化は、TensorBoardの振る舞いと一致させる必要がない場合、標準的な選択肢として優れています。指数移動平均とは異なり、点は値の前後に発生する点に基づいて平滑化されます。

アプリでの表示例はこちらです。[in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

![](/images/app_ui/gaussian_smoothing.png)

## 移動平均

移動平均は、与えられたx値の前後のウィンドウ内の点の平均で点を置き換える平滑化アルゴリズムです。[https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)の「ボックスカーフィルタ」を参照してください。移動平均の選ばれたパラメータは、Weights and Biasesに考慮する移動平均の点の数を伝えます。

x軸上の点が不均一に配置されている場合は、ガウス平滑化を検討してください。

以下の画像は、アプリでの移動平均の表示例を示しています。[in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

![](/images/app_ui/running_average.png)

## 指数移動平均 (非推奨)

> TensorBoardのEMAアルゴリズムは、同じチャート上に一貫した点密度（1単位のx軸あたりの点の数）を持たない複数の線を正確に平滑化できないため、非推奨とされました。

指数移動平均は、TensorBoardの平滑化アルゴリズムと一致するように実装されています。範囲は0から1までです。背景については[指数平滑化](https://www.wikiwand.com/en/Exponential_smoothing)を参照してください。時系列の初期値がゼロに偏らないようにデバイアス項が追加されています。

以下は、このアルゴリズムがどのように動作するかについてのサンプルコードです。

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

アプリでの表示例はこちらです。[in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/exponential_moving_average.png)

## 実装の詳細

すべての平滑化アルゴリズムはサンプルデータに対して実行されます。つまり、1500ポイント以上をログする場合、平滑化アルゴリズムはサーバーからポイントがダウンロードされた後で実行されます。平滑化アルゴリズムの目的は、データのパターンを素早く見つけることです。記録されたポイントが多いメトリクスに対して正確な平滑化値が必要な場合は、APIを通じてメトリクスをダウンロードし、独自の平滑化メソッドを実行する方が良いかもしれません。

## 元データを隠す

デフォルトでは、元の平滑化されていないデータを背景の淡い線として表示します。**Show Original** トグルをクリックしてこれをオフにします。

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)