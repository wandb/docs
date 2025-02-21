---
title: What formula do you use for your smoothing algorithm?
menu:
  support:
    identifier: ja-support-formula_smoothing_algorithm
tags:
- tensorboard
toc_hide: true
type: docs
---

エクスポネンシャル移動平均の式は、TensorBoard で使用されているものと一致します。

同等の Python 実装の詳細については、この [Stack OverFlow の説明](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930) を参照してください。TensorBoard のスムージングアルゴリズムのソース コード（この執筆時点）は[こちら](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699) で見つけることができます。