---
title: Manage user settings
description: 사용자 설정 에서 프로필 정보, 계정 기본값, 알림, 베타 제품 참여, GitHub 인테그레이션 , 저장소 사용량, 계정 활성화를
  관리하고 팀 을 만드세요.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

사용자 프로필 페이지로 이동하여 오른쪽 상단 모서리에 있는 사용자 아이콘을 선택하세요. 드롭다운 메뉴에서 **설정**을 선택합니다.

## 프로필

**프로필** 섹션에서는 계정 이름과 소속 기관을 관리하고 수정할 수 있습니다. 선택적으로 자기소개, 위치, 개인 또는 소속 기관 웹사이트 링크를 추가하고 프로필 이미지를 업로드할 수 있습니다.

## 자기 소개 편집

자기 소개를 편집하려면 프로필 상단의 **편집**을 클릭합니다. 열리는 WYSIWYG 편집기는 Markdown을 지원합니다.
1. 줄을 편집하려면 해당 줄을 클릭합니다. 시간을 절약하기 위해 `/`를 입력하고 목록에서 Markdown을 선택할 수 있습니다.
1. 항목의 드래그 핸들을 사용하여 이동합니다.
1. 블록을 삭제하려면 드래그 핸들을 클릭한 다음 **삭제**를 클릭합니다.
1. 변경 사항을 저장하려면 **저장**을 클릭합니다.

### 소셜 배지 추가

X에서 `@weights_biases` 계정에 대한 팔로우 배지를 추가하려면 배지 이미지를 가리키는 HTML `<img>` 태그가 있는 Markdown 스타일 링크를 추가할 수 있습니다.

```markdown
[<img src="https://img.shields.io/twitter/follow/weights_biases?style=social" alt="X: @weights_biases" >](https://x.com/intent/follow?screen_name=weights_biases)
```
`<img>` 태그에서 `width`, `height` 또는 둘 다 지정할 수 있습니다. 둘 중 하나만 지정하면 이미지 비율이 유지됩니다.

## 팀

**팀** 섹션에서 새 팀을 만듭니다. 새 팀을 만들려면 **새 팀** 버튼을 선택하고 다음을 제공합니다.

* **팀 이름** - 팀의 이름입니다. 팀 이름은 고유해야 합니다. 팀 이름은 변경할 수 없습니다.
* **팀 유형** - **업무** 또는 **학술** 버튼을 선택합니다.
* **회사/조직** - 팀의 회사 또는 조직 이름을 제공합니다. 드롭다운 메뉴를 선택하여 회사 또는 조직을 선택합니다. 선택적으로 새 조직을 제공할 수 있습니다.

{{% alert %}}
관리 계정만 팀을 만들 수 있습니다.
{{% /alert %}}

## 베타 기능

**베타 기능** 섹션에서는 선택적으로 재미있는 추가 기능과 개발 중인 새 제품의 미리 보기를 활성화할 수 있습니다. 활성화하려는 베타 기능 옆에 있는 토글 스위치를 선택합니다.

## 알림

[wandb.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})을 사용하여 run이 충돌하거나 완료될 때 알림을 받고 사용자 정의 알림을 설정합니다. 이메일 또는 Slack을 통해 알림을 받습니다. 알림을 받을 이벤트 유형 옆에 있는 스위치를 토글합니다.

* **Runs finished**: Weights & Biases run이 성공적으로 완료되었는지 여부.
* **Run crashed**: run이 완료되지 못한 경우 알림.

알림을 설정하고 관리하는 방법에 대한 자세한 내용은 [wandb.alert로 알림 보내기]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참조하세요.

## 개인 GitHub 인테그레이션

개인 Github 계정을 연결합니다. Github 계정을 연결하려면:

1. **Github 연결** 버튼을 선택합니다. 그러면 OAuth (Open Authorization) 페이지로 리디렉션됩니다.
2. **조직 엑세스** 섹션에서 엑세스 권한을 부여할 조직을 선택합니다.
3. **wandb**를 **승인**합니다.

## 계정 삭제

**계정 삭제** 버튼을 선택하여 계정을 삭제합니다.

{{% alert color="secondary" %}}
계정 삭제는 되돌릴 수 없습니다.
{{% /alert %}}

## 저장 공간

**저장 공간** 섹션에서는 계정이 Weights & Biases 서버에서 사용한 총 메모리 사용량을 설명합니다. 기본 저장 공간 플랜은 100GB입니다. 저장 공간 및 가격 책정에 대한 자세한 내용은 [가격](https://wandb.ai/site/pricing) 페이지를 참조하세요.
