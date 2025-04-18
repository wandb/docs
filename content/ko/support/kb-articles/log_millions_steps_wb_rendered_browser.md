---
title: What happens when I log millions of steps to W&B? How is that rendered in the
  browser?
menu:
  support:
    identifier: ko-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

전송된 포인트 수는 UI에서 그래프 로딩 시간에 영향을 미칩니다. 1,000포인트를 초과하는 라인의 경우, 백엔드는 데이터를 1,000포인트로 샘플링한 후 브라우저로 전송합니다. 이 샘플링은 비결정적이며 페이지를 새로 고침할 때마다 다른 샘플링된 포인트가 생성됩니다.

메트릭당 10,000포인트 미만으로 기록하세요. 라인에서 100만 포인트 이상을 기록하면 페이지 로드 시간이 크게 늘어납니다. 이 [Colab](http://wandb.me/log-hf-colab)에서 정확도를 희생하지 않고 로깅 공간을 최소화하는 전략을 살펴보세요. 500개 이상의 config 및 요약 메트릭 열이 있는 경우, 테이블에 500개만 표시됩니다.
