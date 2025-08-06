---
title: 여러 아키텍처와 run에서 Artifacts를 사용하시나요?
menu:
  support:
    identifier: ko-support-kb-articles-artifacts_multiple_architectures_runs
support:
- Artifacts
toc_hide: true
type: docs
url: /support/:filename
---

모델 버전 관리를 하는 방법에는 여러 가지가 있습니다. Artifacts는 특정한 요구에 맞춰 모델 버전 관리 기능을 제공합니다. 여러 모델 아키텍처를 실험하는 프로젝트라면 아키텍처별로 artifact를 분리해서 관리하는 것이 일반적입니다. 아래 단계를 참고해 주세요:

1. 각 모델 아키텍처마다 새로운 artifact를 생성하세요. 아키텍처에 대한 추가 정보를 넣으려면 artifact의 `metadata` 속성을 활용하면 됩니다. 이것은 run의 `config`를 사용하는 것과 비슷하게 작동합니다.
2. 각 모델에 대해 주기적으로 checkpoint를 `log_artifact`로 기록하세요. W&B는 이 checkpoint들의 히스토리를 자동으로 관리하며, 가장 최근 것을 `latest` 에일리어스로 표시합니다. 원하는 모델 아키텍처의 최신 checkpoint는 `architecture-name:latest`처럼 참조할 수 있습니다.