---
title: 팀 관리
description: 동료들과 협업하고, 결과를 공유하며, 팀의 모든 실험을 추적하세요.
menu:
  default:
    identifier: ko-guides-models-app-settings-page-teams
    parent: settings
weight: 50
---

W&B Teams를 ML 팀의 중앙 워크스페이스로 활용하여 더 나은 모델을 더 빠르게 구축하세요.

* **팀이 시도한 모든 실험을 추적**하여 중복 작업을 방지할 수 있습니다.
* **이전에 학습된 모델을 저장하고 재현**할 수 있습니다.
* **진행 상황과 결과를** 상사 및 협업자와 쉽게 공유할 수 있습니다.
* **성능 회귀를 즉시 감지**하고 성능 저하 시 즉시 알림을 받을 수 있습니다.
* **모델 성능을 벤치마킹**하고 모델 버전을 비교할 수 있습니다.

{{< img src="/images/app_ui/teams_overview.webp" alt="Teams workspace overview" >}}

## 협업 팀 만들기

1. [회원가입 또는 로그인](https://app.wandb.ai/login?signup=true)하여 무료 W&B 계정을 생성하세요.
2. 네비게이션 바에서 **Invite Team**을 클릭하세요.
3. 팀을 만들고 협업자를 초대하세요.
4. 팀 설정은 [팀 설정 관리]({{< relref path="team-settings.md#privacy" lang="ko" >}})를 참고하세요.

{{% alert %}}
**안내**: 조직의 관리자인 경우에만 새 팀을 생성할 수 있습니다.
{{% /alert %}}

## 팀 프로필 만들기

팀의 프로필 페이지를 커스터마이즈해서 소개글을 작성하고, 공개 또는 팀원에게 보이는 Reports와 Projects를 보여줄 수 있습니다. 또한 외부 링크를 통해 Reports, Projects, 외부 웹사이트도 함께 소개할 수 있습니다.

* **최고의 연구 결과**를 방문자에게 공개하여 팀의 대표 리포트를 강조할 수 있습니다.
* **가장 활발한 프로젝트를 소개**하여 팀원이 쉽게 찾을 수 있도록 할 수 있습니다.
* **외부 링크 추가로 협업자 찾기**—회사나 연구실 홈페이지, 발표 논문 등을 링크로 연결할 수 있습니다.

## 팀 멤버 제거

팀 관리자는 팀 설정 페이지에서 떠나는 멤버 이름 옆의 삭제 버튼을 클릭하여 팀원을 삭제할 수 있습니다. 사용자 탈퇴 후에도 해당 사용자가 기록한 run은 팀에 남습니다.

## 팀 역할 및 권한 관리
팀에 동료를 초대할 때 역할을 선택할 수 있습니다. 팀 역할은 다음과 같습니다:

- **Admin**: 팀 관리자는 다른 관리자나 팀원을 추가/제거할 수 있습니다. 모든 프로젝트를 수정하고 전체 삭제 권한(예: Runs, Projects, Artifacts, Sweeps 삭제 등 포함)을 가집니다.
- **Member**: 일반 팀원입니다. 기본적으로 팀원 초대는 관리자만 가능합니다. 이 설정을 변경하려면 [팀 설정 관리]({{< relref path="team-settings.md#privacy" lang="ko" >}})를 참고하세요.

팀원은 자신이 만든 run만 삭제할 수 있습니다. 예를 들어, 멤버 A와 B가 있다고 할 때, 멤버 B가 팀 B의 Project에서 멤버 A가 소유한 다른 Project로 run을 이동시켰다면, 멤버 A는 B가 이동한 run을 삭제할 수 없습니다. 단, 관리자는 모든 팀원의 run 및 sweep run을 관리할 수 있습니다.
- **View-Only (엔터프라이즈 전용 기능)**: View-Only 멤버는 팀 내에서 run, report, workspace 등 자산을 볼 수 있습니다. report를 팔로우하고 댓글 작성은 가능하지만, project 개요, report, run 생성/수정/삭제는 불가합니다.
- **Custom roles (엔터프라이즈 전용 기능)**: Custom roles를 사용하면 조직 관리자가 **View-Only**나 **Member** 역할을 기반으로 추가 권한을 포함하는 새 역할을 만들 수 있습니다. 팀 관리자는 이런 custom 역할을 팀 멤버에게 할당할 수 있습니다. 자세한 사항은 [Introducing Custom Roles for W&B Teams](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) 를 참고하세요.
- **Service accounts (엔터프라이즈 전용 기능)**: [서비스 계정으로 워크플로우 자동화하기]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ko" >}})를 참고하세요.

{{% alert %}}
한 팀에 관리자(admin)가 2명 이상 있도록 설정하는 것을 권장합니다. 주 관리자가 부재 시 관리 기능이 지속될 수 있는 최선의 방법입니다.
{{% /alert %}}

### 팀 설정
팀 설정에서는 팀 및 멤버의 설정을 관리할 수 있습니다. 이 권한들을 이용해 W&B 내에서 팀을 효과적으로 조직하고 운영할 수 있습니다.

| 권한                   | View-Only | Team Member | Team Admin | 
| ---------------------- | --------- | ----------- | ---------- |
| 팀원 추가              |           |             |     X      |
| 팀원 제거              |           |             |     X      |
| 팀 설정 관리           |           |             |     X      |

### Registry
다음 표는 해당 팀 내 모든 프로젝트에 적용되는 Registry 관련 권한을 정리한 것입니다.

| 권한                        | View-Only | Team Member | Registry Admin | Team Admin | 
| ----------------------------| --------- | ----------- | -------------- | ---------- |
| 에일리어스 추가             |           | X           | X              | X |
| Registry에 모델 추가        |           | X           | X              | X |
| Registry 내 모델 보기        | X         | X           | X              | X |
| 모델 다운로드                | X         | X           | X              | X |
| Registry 관리자 추가/제거     |           |             | X              | X | 
| Protected Alias 추가/제거    |           |             | X              |   | 

Protected alias에 대한 자세한 내용은 [Registry 액세스 컨트롤]({{< relref path="/guides/core/registry/model_registry/access_controls.md" lang="ko" >}})을 참고하세요.

### Reports
Report 권한은 team 내 Reports의 생성, 조회, 편집 권한을 제공합니다. 아래 표는 팀 전체 report에 대한 권한을 요약한 것입니다.

| 권한              | View-Only | Team Member                                     | Team Admin | 
| ----------------- | --------- | ----------------------------------------------- | ---------- |
| Report 보기       | X         | X                                               | X          |
| Report 생성       |           | X                                               | X          |
| Report 편집       |           | X (자신의 report만 편집 가능)                   | X          |
| Report 삭제       |           | X (자신의 report만 삭제 가능)                   | X          |

### Experiments
아래 표는 팀 내 모든 Experiment에 대한 권한을 나타냅니다.

| 권한                                                                                      | View-Only | Team Member | Team Admin | 
| ----------------------------------------------------------------------------------------- | --------- | ----------- | ---------- |
| 실험 메타데이터 보기 (히스토리 메트릭, 시스템 메트릭, 파일, 로그 포함)                    | X         | X           | X          |
| 실험 패널 및 워크스페이스 편집                                                             |           | X           | X          |
| 실험 로그 기록                                                                            |           | X           | X          |
| 실험 삭제                                                                                 |           | X (자신이 생성한 실험만 삭제 가능) |  X  |
| 실험 중단                                                                                 |           | X (자신이 생성한 실험만 중단 가능) |  X  |

### Artifacts
아래 표는 팀 내 모든 Artifacts에 대한 권한을 나타냅니다.

| 권한                 | View-Only | Team Member | Team Admin | 
| -------------------- | --------- | ----------- | ---------- |
| Artifacts 보기       | X         | X           | X          |
| Artifacts 생성       |           | X           | X          |
| Artifacts 삭제       |           | X           | X          |
| 메타데이터 편집      |           | X           | X          |
| 에일리어스 편집      |           | X           | X          |
| 에일리어스 삭제      |           | X           | X          |
| Artifacts 다운로드   |           | X           | X          |

### 시스템 설정 (W&B Server 전용)
시스템 권한을 사용하면 팀 및 멤버 생성·관리는 물론 시스템 설정도 조정할 수 있습니다. 이 권한으로 W&B 인스턴스를 효율적으로 운영할 수 있습니다.

| 권한                      | View-Only | Team Member | Team Admin | System Admin | 
| ------------------------- | --------- | ----------- | ---------- | ------------ |
| 시스템 설정 구성           |           |             |            | X            |
| 팀 생성/삭제              |           |             |            | X            |

### 팀 서비스 계정 동작 방식

* 트레이닝 환경에서 팀을 설정한 경우, 해당 팀의 service account를 사용해 팀 내 private, public 프로젝트에 run 로그를 남길 수 있습니다. 환경 변수에 **WANDB_USERNAME** 또는 **WANDB_USER_EMAIL**이 존재하고 지정된 사용자가 해당 팀 멤버라면, run을 해당 사용자에게 귀속할 수 있습니다.
* 트레이닝 환경에서 팀을 **설정하지 않은 상태**로 service account를 사용하면, run은 서비스 계정의 소속 팀 내 명시한 프로젝트에 기록됩니다. 이 경우에도 **WANDB_USERNAME** 또는 **WANDB_USER_EMAIL**이 환경에 있고 지정 유저가 해당 팀 멤버이면 run이 그 유저에게 귀속됩니다.
* 서비스 계정은 소속(Parent) 팀이 아닌 다른 팀의 private 프로젝트에는 run을 기록할 수 없습니다. 오직 프로젝트 visibility가 `Open`인 경우에만 가능하며, 소속팀 프로젝트에만 서비스 계정으로 로그를 남길 수 있습니다.

## 팀 트라이얼

W&B 요금제 관련 자세한 내용은 [pricing page](https://wandb.ai/site/pricing)를 참고하세요. 모든 데이터는 언제든 대시보드 UI 혹은 [Export API]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})로 다운로드할 수 있습니다.

## 개인정보 설정

팀 설정 페이지에서 모든 팀 프로젝트의 개인정보 설정을 확인할 수 있습니다:
`app.wandb.ai/teams/your-team-name`

## 고급 설정

### Secure storage connector

팀 수준 Secure Storage Connector를 사용하면 팀별로 자체 클라우드 스토리지 버킷을 W&B와 연동할 수 있습니다. 이는 기밀 데이터나 엄격한 규정 준수 요건이 필요한 경우 더욱 강력한 데이터 엑세스 제어 및 데이터 분리를 제공합니다. 자세한 내용은 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 참고하세요.