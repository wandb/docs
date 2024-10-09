---
title: Manage user settings
description: 당신의 사용자 설정에서 프로필 정보, 계정 기본값, 알림, 베타 제품 참여, GitHub 인테그레이션, 저장소 사용량, 계정 활성화 관리 및 팀을 생성하세요.
displayed_sidebar: default
---

사용자 프로필 페이지로 이동하여 오른쪽 상단 모서리에 있는 사용자 아이콘을 선택하세요. 드롭다운 메뉴에서 **Settings**를 선택합니다.

### Profile

**Profile** 섹션에서는 계정 이름과 기관을 관리하고 수정할 수 있습니다. 선택적으로 전기, 위치, 개인 또는 기관 웹사이트 링크를 추가하고 프로필 이미지를 업로드할 수 있습니다.

### Teams

**Team** 섹션에서 새로운 팀을 생성할 수 있습니다. 팀을 새로 만들려면 **New team** 버튼을 선택하고 다음을 입력하세요:

* **Team name** - 팀 이름입니다. 팀 이름은 고유해야 합니다. 팀 이름은 변경할 수 없습니다.
* **Team type** - **Work** 또는 **Academic** 버튼 중 하나를 선택하세요.
* **Company/Organization** - 팀의 회사 또는 조직 이름을 입력하세요. 드롭다운 메뉴를 선택하여 회사 또는 조직을 선택하세요. 새로운 조직을 제공할 수도 있습니다.

:::info
관리자 계정만 팀을 생성할 수 있습니다.
:::

### Beta features

**Beta Features** 섹션에서는 개발 중인 새로운 제품의 재미있는 추가 기능과 미리보기를 선택적으로 활성화할 수 있습니다. 활성화하려는 베타 기능 옆의 토글 스위치를 선택하세요.

### Alerts

Your runs이 실패하거나 완료되었을 때 또는 사용자 정의 경고를 설정할 때 [wandb.alert()](../../runs/alert.md)로 알림을 받습니다. 이메일 또는 Slack을 통해 알림을 받을 수 있습니다. 알림을 받고 싶은 이벤트 유형 옆의 스위치를 전환하세요.

* **Runs finished**: Weights and Biases run이 성공적으로 완료되었는지 여부.
* **Run crashed**: run이 완료되지 못했을 경우 알림.

경고 설정 및 관리에 대한 자세한 내용은 [wandb.alert로 경고 보내기](../../runs/alert.md)를 참조하세요.

### Personal GitHub integration

개인 GitHub 계정을 연결하십시오. GitHub 계정을 연결하려면:

1. **Connect Github** 버튼을 선택하세요. 이는 오픈 인증(OAuth) 페이지로 리디렉션됩니다.
2. **Organization access** 섹션에서 엑세스를 부여할 조직을 선택하세요.
3. **Authorize** **wandb**를 선택합니다.

### Delete your account

계정을 삭제하려면 **Delete Account** 버튼을 선택하세요.

:::caution
계정을 삭제하면 복구할 수 없습니다.
:::

### Storage

**Storage** 섹션에서는 Weights and Biases 서버에서 계정이 사용한 총 메모리 사용량을 설명합니다. 기본 저장소 계획은 100GB입니다. 저장소 및 가격 책정에 대한 자세한 내용은 [Pricing](https://wandb.ai/site/pricing) 페이지를 참조하세요.