---
title: Are there best practices for using Launch effectively?
menu:
  launch:
    identifier: ko-launch-launch-faq-best_practices_launch_effectively
    parent: launch-faq
---

1. 쉬운 설정을 위해 에이전트를 시작하기 전에 큐를 생성하세요. 이렇게 하지 않으면 큐가 추가될 때까지 에이전트가 작동하지 못하게 하는 오류가 발생합니다.

2. 개인 사용자 계정에 연결되지 않도록 W&B 서비스 계정을 만들어 에이전트를 시작하세요.

3. `wandb.config`를 사용하여 하이퍼파라미터를 관리하고 작업 재실행 중에 덮어쓸 수 있습니다. argparse 사용에 대한 자세한 내용은 [이 가이드]({{< relref path="/guides/models/track/config/#set-the-configuration-with-argparse" lang="ko" >}})를 참조하세요.
