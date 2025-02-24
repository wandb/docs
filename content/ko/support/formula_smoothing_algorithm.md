---
title: What formula do you use for your smoothing algorithm?
menu:
  support:
    identifier: ko-support-formula_smoothing_algorithm
tags:
- tensorboard
toc_hide: true
type: docs
---

지수 이동 평균 공식은 TensorBoard에서 사용되는 공식과 일치합니다.

해당하는 Python 구현에 대한 자세한 내용은 [Stack OverFlow에 대한 설명](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930)을 참조하십시오. TensorBoard의 스무딩 알고리즘에 대한 소스 코드는 [여기](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699)에서 찾을 수 있습니다. (본 문서 작성 시점 기준)
