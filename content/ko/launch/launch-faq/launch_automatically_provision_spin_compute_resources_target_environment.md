---
title: Can Launch automatically provision (and spin down) compute resources for me
  in the target environment?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_automatically_provision_spin_compute_resources_target_environment
    parent: launch-faq
---

이 프로세스는 환경에 따라 달라집니다. 리소스는 Amazon SageMaker 및 Vertex에서 프로비저닝됩니다. Kubernetes에서는 autoscaler가 수요에 따라 자동으로 리소스를 조정합니다. W&B의 솔루션 설계자는 재시도, autoscaling, 스팟 인스턴스 노드 풀 사용을 가능하게 하도록 Kubernetes 인프라를 구성하는 데 도움을 줍니다. 지원이 필요하면 support@wandb.com으로 문의하거나 공유된 Slack 채널을 이용하세요.
