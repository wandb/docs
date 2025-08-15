---
title: Kubernetes 에서 에이전트가 필요로 하는 권한은 무엇인가요?
menu:
  launch:
    identifier: ko-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

아래의 Kubernetes 매니페스트는 `wandb` 네임스페이스에 `wandb-launch-agent`라는 역할(role)을 생성합니다. 이 역할은 에이전트가 `wandb` 네임스페이스에서 파드, configmap, secret을 생성하고 파드 로그에 엑세스할 수 있도록 허용합니다. `wandb-cluster-role`은 에이전트가 지정된 모든 네임스페이스에서 파드 생성, 파드 로그 엑세스, secret 및 job 생성, job 상태 확인을 할 수 있도록 합니다.