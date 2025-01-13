---
title: "What formula do you use for your smoothing algorithm?"
toc_hide: true
type: docs
tags:
   - tensorboard
---
The exponential moving average formula aligns with the one used in TensorBoard. 

Refer to this [explanation on Stack OverFlow](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930) for details on the equivalent Python implementation. The source code to TensorBoard's smoothing algorithm (as of this writing) can be found [here](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699).