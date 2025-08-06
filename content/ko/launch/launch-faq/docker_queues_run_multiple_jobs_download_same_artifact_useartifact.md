---
title: 여러 Docker 큐 작업에서 동일한 artifact를 다운로드할 때, 캐싱이 사용되나요? 아니면 매번 run마다 다시 다운로드하나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

캐싱은 존재하지 않습니다. 각 Launch 작업은 독립적으로 실행됩니다. 큐 설정에서 Docker 인수를 사용해 큐 또는 에이전트가 공유 캐시를 마운트하도록 구성하세요.

또한, 특정 유스 케이스의 경우 W&B Artifacts 캐시를 지속적인 볼륨으로 마운트할 수 있습니다.