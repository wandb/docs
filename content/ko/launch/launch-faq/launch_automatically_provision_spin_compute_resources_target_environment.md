---
title: Launch가 대상 환경에서 컴퓨팅 리소스를 자동으로 프로비저닝(생성 및 종료)할 수 있나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-launch_automatically_provision_spin_compute_resources_target_environment
    parent: launch-faq
---

이 프로세스는 환경에 따라 다릅니다. Amazon SageMaker와 Vertex에서는 리소스가 프로비저닝됩니다. Kubernetes에서는 autoscaler 가 수요에 따라 리소스를 자동으로 조정합니다. W&B 의 솔루션 아키텍트가 Kubernetes 인프라를 설정하여 재시도, 오토스케일링, spot 인스턴스 노드 풀 사용을 지원해 드립니다. 지원이 필요하신 경우 support@wandb.com 으로 문의하거나 공유된 Slack 채널을 이용해 주세요.