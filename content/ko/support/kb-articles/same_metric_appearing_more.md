---
title: Why is the same metric appearing more than once?
menu:
  support:
    identifier: ko-support-kb-articles-same_metric_appearing_more
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: same_metric_appearing_more
---
동일한 키로 다양한 데이터 유형을 로깅할 때 데이터베이스에서 분할합니다. 이렇게 하면 UI 드롭다운에 동일한 메트릭 이름의 항목이 여러 개 표시됩니다. 그룹화된 데이터 유형은 `number`, `string`, `bool`, `other`(주로 배열) 및 `Histogram` 또는 `Image`와 같은 모든 `wandb` 데이터 유형입니다. 이 문제를 방지하려면 키당 하나의 유형만 보내세요.

메트릭 이름은 대소문자를 구분하지 않습니다. `"My-Metric"` 및 `"my-metric"`과 같이 대소문자만 다른 이름은 사용하지 마세요.
