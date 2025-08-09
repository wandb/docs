---
title: 작업이나 자동화에 대해 비밀 정보를 지정할 수 있나요? 예를 들어, 사용자에게 직접 보이지 않도록 하고 싶은 API 키 같은 경우요?
menu:
  launch:
    identifier: ko-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

네, 다음 단계를 따라 진행하세요:

1. Kubernetes에서 해당 namespace에 run 을 위한 시크릿을 다음 코맨드로 생성하세요:  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. 시크릿을 생성한 후, run 이 시작될 때 큐에서 해당 시크릿을 주입하도록 설정하세요. 클러스터 관리자는 시크릿을 볼 수 있지만, 최종 사용자들은 볼 수 없습니다.