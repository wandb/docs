---
title: Manage teams
description: 동료와 협업하고, 결과를 공유하며, 팀 전체의 모든 실험을 추적하세요.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-teams
    parent: settings
weight: 50
---

W&B Teams를 사용하여 ML 팀의 중앙 워크스페이스를 구축하여 더 나은 모델을 더 빠르게 만드세요.

* 팀에서 시도한 모든 실험을 추적하여 작업이 중복되지 않도록 하세요.
* 이전에 트레이닝된 모델을 저장하고 재현하세요.
* 상사 및 협력자와 진행 상황과 결과를 공유하세요.
* 회귀를 포착하고 성능이 저하되면 즉시 알림을 받으세요.
* 모델 성능을 벤치마킹하고 모델 버전을 비교하세요.

{{< img src="/images/app_ui/teams_overview.webp" alt="" >}}

## 협업 팀 만들기

1. 무료 W&B 계정에 [**가입하거나 로그인**](https://app.wandb.ai/login?signup=true)하세요.
2. 네비게이션 바에서 **팀 초대**를 클릭하세요.
3. 팀을 만들고 협력자를 초대하세요.
4. 팀을 구성하려면 [팀 설정 관리]({{< relref path="team-settings.md#privacy" lang="ko" >}})를 참조하세요.

{{% alert %}}
**참고**: 조직의 관리자만 새 팀을 만들 수 있습니다.
{{% /alert %}}

## 팀 프로필 만들기

팀 프로필 페이지를 사용자 정의하여 소개를 표시하고 공개 또는 팀 멤버에게 보이는 리포트 및 프로젝트를 소개할 수 있습니다. 리포트, 프로젝트 및 외부 링크를 제시하세요.

* 최고의 공개 리포트를 소개하여 방문자에게 최고의 연구 결과를 강조하세요.
* 팀원이 더 쉽게 찾을 수 있도록 가장 활발한 프로젝트를 소개하세요.
* 회사 또는 연구실 웹사이트 및 게시한 논문에 외부 링크를 추가하여 협력자를 찾으세요.

## 팀 멤버 제거

팀 관리자는 팀 설정 페이지를 열고 떠나는 멤버의 이름 옆에 있는 삭제 버튼을 클릭할 수 있습니다. 사용자가 떠난 후에도 팀에 기록된 모든 run은 유지됩니다.

## 팀 역할 및 권한 관리
동료를 팀에 초대할 때 팀 역할을 선택하세요. 다음과 같은 팀 역할 옵션이 있습니다.

- **관리자**: 팀 관리자는 다른 관리자나 팀 멤버를 추가하거나 제거할 수 있습니다. 모든 프로젝트를 수정할 수 있는 권한과 완전한 삭제 권한이 있습니다. 여기에는 run, 프로젝트, 아티팩트 및 스윕 삭제가 포함되지만 이에 국한되지는 않습니다.
- **멤버**: 팀의 일반 멤버입니다. 기본적으로 관리자만 팀 멤버를 초대할 수 있습니다. 이 동작을 변경하려면 [팀 설정 관리]({{< relref path="team-settings.md#privacy" lang="ko" >}})를 참조하세요.

팀 멤버는 자신이 만든 run만 삭제할 수 있습니다. 멤버 A와 B가 있다고 가정합니다. 멤버 B가 팀 B의 프로젝트에서 멤버 A가 소유한 다른 프로젝트로 run을 이동합니다. 멤버 A는 멤버 B가 멤버 A의 프로젝트로 이동한 run을 삭제할 수 없습니다. 관리자는 모든 팀 멤버가 만든 run과 스윕 run을 관리할 수 있습니다.
- **보기 전용 (엔터프라이즈 전용 기능)**: 보기 전용 멤버는 run, 리포트 및 워크스페이스와 같은 팀 내 자산을 볼 수 있습니다. 리포트를 팔로우하고 댓글을 달 수 있지만 프로젝트 개요, 리포트 또는 run을 생성, 편집 또는 삭제할 수는 없습니다.
- **사용자 정의 역할 (엔터프라이즈 전용 기능)**: 사용자 정의 역할을 사용하면 조직 관리자가 세분화된 엑세스 제어를 위해 추가 권한과 함께 **보기 전용** 또는 **멤버** 역할 중 하나를 기반으로 새 역할을 구성할 수 있습니다. 그런 다음 팀 관리자는 해당 사용자 정의 역할을 각 팀의 사용자에게 할당할 수 있습니다. 자세한 내용은 [W&B 팀을 위한 사용자 정의 역할 소개](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3)를 참조하세요.
- **서비스 계정 (엔터프라이즈 전용 기능)**: [서비스 계정을 사용하여 워크플로우 자동화]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ko" >}})를 참조하세요.

{{% alert %}}
W&B는 팀에 둘 이상의 관리자를 두는 것을 권장합니다. 기본 관리자를 사용할 수 없을 때 관리자 작업이 계속될 수 있도록 하는 것이 가장 좋습니다.
{{% /alert %}}

### 팀 설정
팀 설정을 사용하면 팀 및 팀 멤버에 대한 설정을 관리할 수 있습니다. 이러한 권한을 통해 W&B 내에서 팀을 효과적으로 감독하고 구성할 수 있습니다.

| 권한              | 보기 전용 | 팀 멤버 | 팀 관리자 |
| ------------------- | --------- | ----------- | ---------- |
| 팀 멤버 추가       |           |             |     X      |
| 팀 멤버 제거       |           |             |     X      |
| 팀 설정 관리       |           |             |     X      |

### 레지스트리
다음 표에는 지정된 팀의 모든 프로젝트에 적용되는 권한이 나와 있습니다.

| 권한                       | 보기 전용 | 팀 멤버 | 레지스트리 관리자 | 팀 관리자 |
| ---------------------------| --------- | ----------- | -------------- | ---------- |
| 에일리어스 추가                 |           | X           | X              | X |
| 레지스트리에 모델 추가        |           | X           | X              | X |
| 레지스트리에서 모델 보기     | X         | X           | X              | X |
| 모델 다운로드              | X         | X           | X              | X |
| 레지스트리 관리자 추가 또는 제거 |           |             | X              | X |
| 보호된 에일리어스 추가 또는 제거 |           |             | X              |   |

보호된 에일리어스에 대한 자세한 내용은 [레지스트리 엑세스 제어]({{< relref path="/guides/core/registry/model_registry/access_controls.md" lang="ko" >}})를 참조하세요.

### 리포트
리포트 권한은 리포트를 생성, 보고 편집할 수 있는 엑세스 권한을 부여합니다. 다음 표에는 지정된 팀의 모든 리포트에 적용되는 권한이 나와 있습니다.

| 권한       | 보기 전용 | 팀 멤버                                     | 팀 관리자 |
| -----------   | --------- | ----------------------------------------------- | ---------- |
| 리포트 보기   | X         | X                                               | X          |
| 리포트 만들기 |           | X                                               | X          |
| 리포트 편집   |           | X (팀 멤버는 자신의 리포트만 편집할 수 있음) | X          |
| 리포트 삭제   |           | X (팀 멤버는 자신의 리포트만 편집할 수 있음) | X          |

### 실험
다음 표에는 지정된 팀의 모든 실험에 적용되는 권한이 나와 있습니다.

| 권한                                                                              | 보기 전용 | 팀 멤버 | 팀 관리자 |
| ------------------------------------------------------------------------------------ | --------- | ----------- | ---------- |
| 실험 메타데이터 보기 (기록 메트릭, 시스템 메트릭, 파일 및 로그 포함) | X         | X           | X          |
| 실험 패널 및 워크스페이스 편집                                                    |           | X           | X          |
| 실험 기록                                                                          |           | X           | X          |
| 실험 삭제                                                                        |           | X (팀 멤버는 자신이 만든 실험만 삭제할 수 있음) |  X  |
| 실험 중지                                                                        |           | X (팀 멤버는 자신이 만든 실험만 중지할 수 있음)   |  X  |

### 아티팩트
다음 표에는 지정된 팀의 모든 아티팩트에 적용되는 권한이 나와 있습니다.

| 권한            | 보기 전용 | 팀 멤버 | 팀 관리자 |
| ---------------- | --------- | ----------- | ---------- |
| 아티팩트 보기     | X         | X           | X          |
| 아티팩트 만들기   |           | X           | X          |
| 아티팩트 삭제     |           | X           | X          |
| 메타데이터 편집  |           | X           | X          |
| 에일리어스 편집   |           | X           | X          |
| 에일리어스 삭제   |           | X           | X          |
| 아티팩트 다운로드|           | X           | X          |

### 시스템 설정 (W&B 서버만 해당)
시스템 권한을 사용하여 팀 및 팀 멤버를 만들고 관리하고 시스템 설정을 조정합니다. 이러한 권한을 통해 W&B 인스턴스를 효과적으로 관리하고 유지 관리할 수 있습니다.

| 권한                  | 보기 전용 | 팀 멤버 | 팀 관리자 | 시스템 관리자 |
| ------------------------ | --------- | ----------- | ---------- | ------------ |
| 시스템 설정 구성       |           |             |            | X            |
| 팀 생성/삭제           |           |             |            | X            |

### 팀 서비스 계정 행동

* 트레이닝 환경에서 팀을 구성할 때 해당 팀의 서비스 계정을 사용하여 해당 팀 내의 비공개 또는 공개 프로젝트에 run을 기록할 수 있습니다. 또한 환경에 **WANDB_USERNAME** 또는 **WANDB_USER_EMAIL** 변수가 있고 참조된 사용자가 해당 팀의 구성원인 경우 해당 run을 사용자에게 귀속시킬 수 있습니다.
* 트레이닝 환경에서 팀을 구성 **하지 않고** 서비스 계정을 사용하는 경우 run은 해당 서비스 계정의 상위 팀 내에서 명명된 프로젝트에 기록됩니다. 이 경우에도 환경에 **WANDB_USERNAME** 또는 **WANDB_USER_EMAIL** 변수가 있고 참조된 사용자가 서비스 계정의 상위 팀의 구성원인 경우 run을 사용자에게 귀속시킬 수 있습니다.
* 서비스 계정은 상위 팀과 다른 팀의 비공개 프로젝트에 run을 기록할 수 없습니다. 프로젝트가 `공개` 프로젝트 가시성으로 설정된 경우에만 서비스 계정이 프로젝트에 run을 기록할 수 있습니다.

## 팀 트라이얼

W&B 요금제에 대한 자세한 내용은 [요금 페이지](https://wandb.ai/site/pricing)를 참조하세요. 대시보드 UI 또는 [내보내기 API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 언제든지 모든 데이터를 다운로드할 수 있습니다.

## 개인 정보 설정

팀 설정 페이지에서 모든 팀 프로젝트의 개인 정보 설정을 확인할 수 있습니다.
`app.wandb.ai/teams/your-team-name`

## 고급 구성

### 보안 스토리지 커넥터

팀 수준 보안 스토리지 커넥터를 사용하면 팀에서 W&B와 함께 자체 클라우드 스토리지 버킷을 사용할 수 있습니다. 이는 매우 민감한 데이터 또는 엄격한 규정 준수 요구 사항이 있는 팀에 대해 더 나은 데이터 엑세스 제어 및 데이터 격리를 제공합니다. 자세한 내용은 [보안 스토리지 커넥터]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 참조하세요.
