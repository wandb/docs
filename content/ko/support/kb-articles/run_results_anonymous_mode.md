---
title: 계정이 없는 사람이 run 결과를 어떻게 볼 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-run_results_anonymous_mode
support:
- 익명
toc_hide: true
type: docs
url: /support/:filename
---

누군가 `anonymous="allow"` 옵션으로 스크립트를 실행하면:

1. **임시 계정 자동 생성**: W&B 는 로그인된 계정이 있는지 확인하고, 없으면 새로운 익명 계정을 생성한 후 해당 세션에 대한 API 키를 저장합니다.
2. **빠른 결과 로그**: 사용자는 스크립트를 반복적으로 실행하여 W&B 대시보드에서 즉시 결과를 확인할 수 있습니다. 이 미인증 익명 run 은 7일 동안 유지됩니다.
3. **필요할 때 데이터 소유**: 사용자가 W&B 에서 유용한 결과를 발견하면, 페이지 상단 배너의 버튼을 클릭하여 해당 run 데이터를 실제 계정으로 저장할 수 있습니다. 소유하지 않으면 run 데이터는 7일 후 삭제됩니다.

{{% alert color="secondary" %}}
**익명 run 링크는 민감합니다**. 이 링크를 가지고 있으면 누구든지 실험 결과를 7일 동안 확인하고 소유할 수 있으니, 신뢰할 수 있는 사람에게만 링크를 공유하세요. 작가의 신원을 숨긴 채 공개적으로 결과를 공유하려면 support@wandb.com 으로 문의해 주세요.
{{% /alert %}}

W&B 사용자가 스크립트를 찾고 실행하면, 그들의 결과는 일반 run 과 같이 계정에 정확히 로그됩니다.