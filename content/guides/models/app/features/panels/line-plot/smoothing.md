---
description: In line plots, use smoothing to see trends in noisy data.
menu:
  default:
    identifier: smoothing
    parent: line-plot
title: Smooth line plots
weight: 30
---

W&B supports several types of smoothing:

- [Time weighted exponential moving average (TWEMA) smoothing]({{< relref "#time-weighted-exponential-moving-average-twema-smoothing-default" >}}) 
- [Gaussian smoothing]({{< relref "#gaussian-smoothing" >}})
- [Running average]({{< relref "#running-average-smoothing" >}})
- [Exponential moving average (EMA) smoothing]({{< relref "#exponential-moving-average-ema-smoothing" >}})

See these live in an [interactive W&B report](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc).

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="Demo of various smoothing algorithms" >}}

## Time Weighted Exponential Moving Average (TWEMA) smoothing (Default)

The Time Weighted Exponential Moving Average (TWEMA) smoothing algorithm is a technique for smoothing time series data by exponentially decaying the weight of previous points. For details about the technique, see [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing). The range is 0 to 1. There is a de-bias term added so that early values in the time series are not biased towards zero.

The TWEMA algorithm takes the density of points on the line (the number of `y` values per unit of range on x-axis) into account. This allows consistent smoothing when displaying multiple lines with different characteristics simultaneously.

Here is sample code for how this works under the hood:

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

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="Demo of TWEMA smoothing" >}}

## Gaussian smoothing

Gaussian smoothing (or gaussian kernel smoothing) computes a weighted average of the points, where the weights correspond to a gaussian distribution with the standard deviation specified as the smoothing parameter. The smoothed value is calculated for every input x value.

Gaussian smoothing is a good standard choice for smoothing if your points or spaced unevenly on the x-axis, or if you are not concerned with matching a different smoothing behavior. Unlike with TWEMA or EMA, the point will be smoothed based on points occurring both before and after the value.

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="Demo of gaussian smoothing" >}}

## Running average smoothing

Running average is a smoothing algorithm that replaces a point with the average of points in a window before and after the given x value. See "Boxcar Filter" at [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average). The selected parameter for running average tells Weights and Biases the number of points to consider in the moving average.

Consider using Gaussian Smoothing instead if your points are spaced unevenly on the x-axis.

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

{{< img src="/images/app_ui/running_average.png" alt="Demo of running average smoothing" >}}

## Exponential Moving Average (EMA) smoothing

The Exponential Moving Average (EMA) smoothing algorithm is a rule of thumb technique for smoothing time series data using the exponential window function. For details about the technique, see [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing). The range is 0 to 1. A debias term is added so that early values in the time series are not biases towards zero.

On a best-effort basis, EMA smoothing is applied at the back end by W&B Server. In this situation, EMA smoothing is applied to a full scan of history, rather than bucketing first before smoothing. In general, this improves the accuracy of the smoothing operation.

In the following situations, EMA smoothing is instead applied at the front end by the W&B App, after bucketing.
- Sampling
- Grouping
- Expressions
- Non-monotonic x-axes
- time-based x-axes

Here is sample code for how this works under the hood:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

Here's what this looks like [in the app](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/exponential_moving_average.png" alt="Demo of EMA smoothing" >}}

## Implementation Details

With the exception of back end EMA smoothing, all of the smoothing algorithms run on the sampled data. If you log more than 1500 points, the smoothing algorithm will run _after_ the points are downloaded from W&B Server. The intention of the smoothing algorithms is to help find patterns in data quickly. If you need exact smoothed values on metrics with a large number of logged points, it may be better to download your metrics through the API and run your own smoothing methods.

## Hide original data

By default, the original, unsmoothed data displays in the plot as a faint line in the background. Click the **Show Original** toggle to turn this off.

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="Turn on or off original data" >}}