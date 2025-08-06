---
title: 수백만 개의 스텝을 W&B에 로그하면 어떻게 되나요? 브라우저에서는 이것이 어떻게 표시되나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

UI에서 그래프를 불러올 때 전송되는 포인트 수에 따라 로딩 시간이 달라집니다. 한 라인에 1,000포인트를 초과하면 백엔드에서 데이터를 1,000포인트로 샘플링한 후 브라우저로 전송합니다. 이 샘플링은 비결정적이어서 페이지를 새로 고침할 때마다 샘플링된 포인트가 다를 수 있습니다.

각 메트릭마다 10,000포인트 미만으로 로그를 남기세요. 한 라인에 100만 포인트 이상을 기록하면 페이지 로드 시간이 크게 증가합니다. [Colab](https://wandb.me/log-hf-colab)에서 정확도를 유지하면서 로그량을 최소화하는 방법을 살펴보세요. config와 summary 메트릭의 컬럼이 500개를 초과하면, 테이블에는 최대 500개만 표시됩니다.