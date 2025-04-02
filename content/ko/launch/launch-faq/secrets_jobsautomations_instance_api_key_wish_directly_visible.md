---
title: Can you specify secrets for jobs/automations? For instance, an API key which
  you do not wish to be directly visible to users?
menu:
  launch:
    identifier: ko-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

예. 다음 단계를 따르세요:

1. 다음 코맨드를 사용하여 run에 지정된 네임스페이스에 Kubernetes secret을 생성합니다:  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. secret을 생성한 후, run이 시작될 때 secret을 삽입하도록 큐를 구성합니다. 클러스터 관리자만 secret을 볼 수 있으며, 최종 사용자는 볼 수 없습니다.
