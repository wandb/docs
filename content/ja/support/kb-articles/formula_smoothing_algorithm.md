---
title: スムージングアルゴリズムにはどのような数式を使用していますか？
menu:
  support:
    identifier: ja-support-kb-articles-formula_smoothing_algorithm
support:
- TensorBoard
toc_hide: true
type: docs
url: /support/:filename
---

指数移動平均の式は、TensorBoard で使われているものと一致します。

同等の Python 実装の詳細は、この [Stack OverFlow の解説](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930) を参照してください。TensorBoard の平滑化アルゴリズムのソースコードは（執筆時点では）[こちら](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699) にあります。