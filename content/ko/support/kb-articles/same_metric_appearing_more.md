---
title: 왜 동일한 metric 이 여러 번 나타나나요?
menu:
  support:
    identifier: ko-support-kb-articles-same_metric_appearing_more
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

동일한 키 아래에 다양한 데이터 유형을 로그할 경우, 데이터베이스에서 이를 분리하게 됩니다. 이로 인해 UI의 드롭다운 메뉴에 동일한 metric 이름이 여러 번 나타날 수 있습니다. 그룹화되는 데이터 유형에는 `number`, `string`, `bool`, `other`(주로 배열), 그리고 `wandb` 데이터 유형(예: `Histogram` 또는 `Image`)이 포함됩니다. 이러한 문제가 발생하지 않도록 하나의 키에는 한 가지 유형만 전송하세요.

Metric 이름은 대소문자를 구분하지 않습니다. `"My-Metric"`과 `"my-metric"`처럼 대소문자만 다른 이름의 사용은 피하세요.