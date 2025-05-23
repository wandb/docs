---
title: Does Launch support parallelization?  How can I limit the resources consumed
  by a job?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch는 여러 GPU 및 노드에서 작업 확장을 지원합니다. 자세한 내용은 [이 가이드]({{< relref path="/launch/integration-guides/volcano.md" lang="ko" >}})를 참조하세요.

각 Launch 에이전트는 실행할 수 있는 최대 동시 작업 수를 결정하는 `max_jobs` 파라미터로 구성됩니다. 여러 에이전트가 적절한 실행 인프라에 연결되어 있는 한 단일 대기열을 가리킬 수 있습니다.

리소스 설정에서 대기열 또는 작업 Run 수준에서 CPU, GPU, 메모리 및 기타 리소스에 대한 제한을 설정할 수 있습니다. Kubernetes에서 리소스 제한을 사용하여 대기열을 설정하는 방법에 대한 자세한 내용은 [이 가이드]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ko" >}})를 참조하세요.

Sweeps의 경우 다음 블록을 대기열 설정에 포함하여 동시 Runs 수를 제한합니다.

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
