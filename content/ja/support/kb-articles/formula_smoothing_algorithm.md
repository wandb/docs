---
title: 平滑化アルゴリズムにはどのような式を使っていますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- tensorboard
---

指数移動平均の式は、TensorBoard で使われているものと一致しています。

同等の Python 実装に関する詳細は、[Stack OverFlow のこの説明](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930) を参照してください。TensorBoard のスムージングアルゴリズムのソースコード（執筆時点）は[こちら](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699)で確認できます。