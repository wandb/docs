---
title: 그래프 평활화 알고리즘에 어떤 수식을 사용하나요?
menu:
  support:
    identifier: ko-support-kb-articles-formula_smoothing_algorithm
support:
- 텐서보드
toc_hide: true
type: docs
url: /support/:filename
---

지수 이동 평균 공식은 TensorBoard에서 사용되는 공식과 일치합니다.

동일한 Python 구현에 대한 자세한 내용은 이 [Stack OverFlow 설명](https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar/75421930#75421930)을 참고하세요. TensorBoard의 스무딩 알고리즘 소스 코드는 (이 문서를 작성하는 시점 기준으로) [여기](https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699)에서 확인할 수 있습니다.