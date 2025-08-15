---
title: W&B 서버의 자동 업데이트 비활성화
description: W&B 서버의 자동 업데이트를 비활성화하는 방법을 알아보세요.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-disable-automatic-app-version-updates
    parent: self-managed
weight: 99
---

이 페이지는 W&B Server의 자동 버전 업그레이드를 비활성화하고 버전을 고정하는 방법을 안내합니다. 이 안내는 [W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ko" >}})로 관리되는 배포 환경에만 적용됩니다.

{{% alert %}}
W&B는 W&B Server의 주요 버전을 최초 출시일로부터 12개월 동안 지원합니다. **셀프 관리** 인스턴스를 사용하는 고객은 지원을 유지하기 위해 제때 업그레이드를 직접 진행해야 합니다. 지원이 중단된 버전에 머무르는 것은 피하세요. W&B는 **셀프 관리** 인스턴스를 사용하는 고객에게 최소 분기별(3개월마다)로 최신 릴리즈로 배포를 업데이트할 것을 강력히 권장합니다. 이렇게 하면 최신 기능, 성능 개선, 그리고 버그 수정 등을 받을 수 있습니다.
{{% /alert %}}

## 요구 사항

- W&B Kubernetes Operator `v1.13.0` 이상
- System Console `v2.12.2` 이상

이 요구 사항을 충족하는지 확인하려면, 인스턴스의 W&B Custom Resource 또는 Helm chart를 참고하세요. `operator-wandb`와 `system-console` 컴포넌트의 `version` 값을 확인하면 됩니다.

## 자동 업데이트 비활성화하기
1. `admin` 역할이 있는 사용자로 W&B App에 로그인합니다.
2. 상단의 사용자 아이콘을 클릭한 뒤, **System Console**을 선택합니다.
3. **Settings** > **Advanced**로 이동한 다음, **Other** 탭을 클릭합니다.
4. **Disable Auto Upgrades** 섹션에서 **Pin specific version**을 활성화합니다.
5. **Select a version** 드롭다운에서 원하는 W&B Server 버전을 선택합니다.
6. **Save**를 클릭합니다.

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="Disable Automatic Updates Saved" >}}

    이제 자동 업그레이드가 비활성화되고, W&B Server가 선택한 버전에 고정됩니다.
1. 자동 업그레이드가 꺼진 것을 확인하려면, **Operator** 탭으로 이동해서 reconciliation 로그에서 문자열 `Version pinning is enabled`를 검색하세요.

```
│info 2025-04-17T17:24:16Z wandb default 변경사항 없음
│info 2025-04-17T17:24:16Z wandb default 활성 스펙을 찾았습니다
│info 2025-04-17T17:24:16Z wandb default 원하는 스펙
│info 2025-04-17T17:24:16Z wandb default 라이선스
│info 2025-04-17T17:24:16Z wandb default Version Pinning is enabled
│info 2025-04-17T17:24:16Z wandb default Weights & Biases 인스턴스를 찾아 스펙을 처리합니다...
│info 2025-04-17T17:24:16Z wandb default === Weights & Biases 인스턴스 동기화 중...
```