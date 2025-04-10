---
title: Using artifacts with multiple architectures and runs?
menu:
  support:
    identifier: ko-support-kb-articles-artifacts_multiple_architectures_runs
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

모델을 버전 관리하는 방법에는 여러 가지가 있습니다. Artifacts는 특정 요구 사항에 맞춘 모델 버전 관리를 위한 툴을 제공합니다. 여러 모델 아키텍처를 탐색하는 프로젝트의 일반적인 접근 방식은 Artifacts를 아키텍처별로 분리하는 것입니다. 다음 단계를 고려하십시오.

1. 각기 다른 모델 아키텍처에 대해 새 아티팩트를 생성합니다. Artifacts의 `metadata` 속성을 사용하여 run의 `config` 사용과 유사하게 아키텍처에 대한 자세한 설명을 제공합니다.
2. 각 모델에 대해 `log_artifact`로 체크포인트를 주기적으로 로그합니다. W&B는 이러한 체크포인트의 기록을 구축하고 가장 최근의 체크포인트에 `latest` 에일리어스를 레이블링합니다. `architecture-name:latest`를 사용하여 모든 모델 아키텍처의 최신 체크포인트를 참조하십시오.
