---
title: What permissions does the agent require in Kubernetes?
menu:
  launch:
    identifier: ko-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

다음 Kubernetes 매니페스트는 `wandb` 네임스페이스에 `wandb-launch-agent` 라는 역할을 생성합니다. 이 역할은 에이전트가 `wandb` 네임스페이스에서 pod, configmap, secrets를 생성하고 pod 로그에 엑세스할 수 있도록 합니다. `wandb-cluster-role` 은 에이전트가 pod를 생성하고, pod 로그에 엑세스하고, secrets, jobs를 생성하고, 지정된 모든 네임스페이스에서 job 상태를 확인할 수 있도록 합니다.
