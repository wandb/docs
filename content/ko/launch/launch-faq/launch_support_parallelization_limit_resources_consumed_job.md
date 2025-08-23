---
title: Launch에서 병렬 처리를 지원하나요? Job이 사용하는 리소스를 제한하려면 어떻게 해야 하나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch 는 여러 GPU 및 노드에 걸쳐 작업을 스케일링할 수 있습니다. 자세한 내용은 [Volcano 인테그레이션 가이드]({{< relref path="/launch/integration-guides/volcano.md" lang="ko" >}})를 참고하세요.

각 launch 에이전트는 동시에 실행할 수 있는 최대 작업 수를 결정하는 `max_jobs` 파라미터로 설정됩니다. 여러 에이전트가 적절한 launching 인프라에 연결되어 있다면 하나의 큐를 함께 사용할 수 있습니다.

CPU, GPU, 메모리 등 다양한 리소스의 제한은 큐 또는 job run 단위의 resource 설정에서 지정할 수 있습니다. Kubernetes 에서 리소스 제한이 있는 큐를 설정하는 방법은 [Kubernetes 설정 가이드]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ko" >}})를 참고하세요.

Sweeps 를 위한 동시 실행 run 개수 제한은 큐 설정에 아래 블록을 추가하면 됩니다.

```yaml title="queue config"
  scheduler:
    num_workers: 4
```