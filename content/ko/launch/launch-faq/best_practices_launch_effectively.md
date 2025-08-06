---
title: Launch를 효과적으로 사용하기 위한 모범 사례가 있나요?
menu:
  launch:
    identifier: ko-launch-launch-faq-best_practices_launch_effectively
    parent: launch-faq
---

1. 에이전트를 시작하기 전에 queue를 생성하면 설정이 더 간편해집니다. queue를 미리 만들지 않으면, queue가 추가될 때까지 에이전트가 작동하지 않는 오류가 발생합니다.

2. 에이전트를 실행하려면 W&B 서비스 계정을 생성하세요. 개인 사용자 계정과 연결되지 않아야 합니다.

3. 하이퍼파라미터 관리를 위해 `wandb.Run.config` 를 사용하세요. 이를 통해 작업을 다시 실행할 때 값을 덮어쓸 수 있습니다. argparse 사용 방법은 [argparse로 설정하기 가이드]({{< relref path="/guides/models/track/config/#set-the-configuration-with-argparse" lang="ko" >}})를 참고하세요.