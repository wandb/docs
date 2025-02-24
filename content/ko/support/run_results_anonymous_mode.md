---
title: How does someone without an account see run results?
menu:
  support:
    identifier: ko-support-run_results_anonymous_mode
tags:
- anonymous
toc_hide: true
type: docs
---

만약 누군가가 `anonymous="allow"`로 스크립트를 실행한다면:

1. **임시 계정 자동 생성**: W&B는 로그인된 계정이 있는지 확인합니다. 만약 없다면, W&B는 새로운 익명 계정을 생성하고 해당 세션의 API 키 를 저장합니다.
2. **결과 를 빠르게 로그**: 사용자는 스크립트 를 반복적으로 실행하고 W&B 대시보드 에서 결과 를 즉시 볼 수 있습니다. 클레임되지 않은 이러한 익명 run 은 7일 동안 사용할 수 있습니다.
3. **유용할 때 데이터 클레임**: 사용자가 W&B에서 가치 있는 결과 를 식별하면 페이지 상단의 배너에 있는 버튼을 클릭하여 run 데이터 를 실제 계정에 저장할 수 있습니다. 클레임하지 않으면 run 데이터 는 7일 후에 삭제됩니다.

{{% alert color="secondary" %}}
**익명 run 링크는 민감합니다**. 이러한 링크를 통해 누구나 7일 동안 실험 결과 를 보고 클레임할 수 있으므로 신뢰할 수 있는 개인과만 링크를 공유하십시오. 작성자의 신원을 숨기면서 결과 를 공개적으로 공유하려면 support@wandb.com 으로 문의하여 지원을 받으십시오.
{{% /alert %}}

W&B 사용자가 스크립트 를 찾아 실행하면 일반 run 과 마찬가지로 결과 가 계정에 올바르게 로그됩니다.
