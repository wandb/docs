---
title: When multiple jobs in a Docker queue download the same artifact, is any caching
  used, or is it re-downloaded every run?
menu:
  launch:
    identifier: ko-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

캐싱은 존재하지 않습니다. 각 Launch 작업은 독립적으로 작동합니다. 대기열 또는 에이전트가 대기열 설정에서 Docker 인수를 사용하여 공유 캐시를 마운트하도록 구성하세요.

또한 특정 유스 케이스에 대해 W&B Artifacts 캐시를 영구 볼륨으로 마운트하세요.
