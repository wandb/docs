---
title: あなたの平滑化アルゴリズムにはどのような式を使用していますか？
menu:
  support:
    identifier: ja-support-kb-articles-formula_smoothing_algorithm
support:
- tensorboard
toc_hide: true
type: docs
url: /support/:filename
---

The exponential moving average formula aligns with the one used in TensorBoard. 

詳細については、この[Stack OverFlow の説明](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930)を参照してください。TensorBoard のスムージング アルゴリズムのソース コード（執筆時点）は[こちら](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699)で見ることができます。