---
title: 사용자 설정 관리
description: 사용자 설정에서 프로필 정보, 계정 기본값, 알림, 베타 제품 참여, GitHub 인테그레이션, 스토리지 사용량, 계정 활성화
  관리 및 팀 생성을 할 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-user-settings
    parent: settings
weight: 10
---

사용자 프로필 페이지로 이동하여 오른쪽 상단의 사용자 아이콘을 선택하세요. 드롭다운에서 **설정**을 선택합니다.

## 프로필

**프로필** 섹션에서는 계정 이름과 소속 기관을 관리하고 수정할 수 있습니다. 바이오, 위치, 개인 또는 소속 기관 웹사이트 링크, 프로필 이미지를 추가할 수 있습니다.

## 소개 수정

소개를 수정하려면, 프로필 상단의 **수정** 버튼을 클릭하세요. 열리는 WYSIWYG 에디터는 Markdown을 지원합니다.
1. 수정할 줄을 클릭하세요. 시간을 절약하려면 `/` 를 입력하고 목록에서 Markdown을 선택할 수 있습니다.
1. 항목의 드래그 핸들을 사용해 순서를 변경하세요.
1. 블록을 삭제하려면 드래그 핸들을 클릭한 다음 **삭제**를 클릭하세요.
1. 변경 사항을 저장하려면 **저장**을 클릭하세요.

### 소셜 배지 추가하기

X의 `@weights_biases` 계정 팔로우 배지를 추가하려면, Markdown 스타일 링크와 배지 이미지가 있는 HTML `<img>` 태그를 사용할 수 있습니다.

```markdown
[![X: @weights_biases](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://x.com/intent/follow?screen_name=weights_biases)
```
`<img>` 태그에서 `width`, `height` 혹은 둘 다 지정할 수 있습니다. 둘 중 하나만 지정하면 이미지 비율이 유지됩니다.

## Teams

**Team** 섹션에서 새 팀을 만들 수 있습니다. 새로운 팀을 만들려면 **새 팀** 버튼을 선택하고 아래 정보를 입력하세요.

* **Team name** - 팀의 이름입니다. 팀 이름은 고유해야 하며, 한 번 정하면 변경할 수 없습니다.
* **Team type** - **Work** 또는 **Academic** 버튼 중에서 선택하세요.
* **Company/Organization** - 팀이 속한 회사 또는 기관 이름을 입력하세요. 드롭다운 메뉴에서 선택하거나 새로운 조직명을 직접 입력할 수 있습니다.

{{% alert %}}
관리자 계정만 팀을 생성할 수 있습니다.
{{% /alert %}}

## 베타 기능

**베타 기능** 섹션에서는 개발 중인 새로운 제품의 추가 기능이나 미리보기를 선택적으로 사용할 수 있습니다. 활성화하려는 베타 기능 옆의 토글 스위치를 선택하세요.

## 알림

run이 크래시되거나 완료될 때 또는 맞춤 알림을 설정할 때 [wandb.Run.alert()]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 통해 알림을 받을 수 있습니다. 이메일 또는 Slack을 통해 알림을 받을 수 있습니다. 원하는 이벤트 유형 옆의 스위치를 켜서 알림을 설정하세요.

* **Runs finished**: Weights and Biases run이 성공적으로 완료되었는지 확인합니다.
* **Run crashed**: run이 정상적으로 종료되지 않은 경우 알림을 받습니다.

알림 설정 및 관리 방법에 대한 자세한 내용은 [wandb.Run.alert()로 알림 보내기]({{< relref path="/guides/models/track/runs/alert.md" lang="ko" >}})를 참고하세요.

## 개인 GitHub 인테그레이션

개인 Github 계정을 연결할 수 있습니다. Github 계정 연결 방법:

1. **Connect Github** 버튼을 선택하세요. 그러면 오픈 인증(OAuth) 페이지로 이동합니다.
2. **Organization access** 섹션에서 엑세스 권한을 부여할 조직을 선택하세요.
3. **wandb**를 **Authorize** 하세요.

## 계정 삭제

**계정 삭제** 버튼을 선택하여 계정을 삭제할 수 있습니다.

{{% alert color="secondary" %}}
계정 삭제는 되돌릴 수 없습니다.
{{% /alert %}}

## 저장소

**저장소** 섹션에서는 Weights and Biases 서버에서 계정이 사용한 총 메모리 용량을 확인할 수 있습니다. 기본 저장소 용량은 100GB입니다. 저장소와 비용에 대한 자세한 내용은 [Pricing](https://wandb.ai/site/pricing) 페이지를 참고하세요.