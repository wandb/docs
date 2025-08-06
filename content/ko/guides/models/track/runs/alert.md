---
title: 알림 보내기
description: Python 코드에서 트리거된 알림을 Slack 또는 이메일로 보내기
menu:
  default:
    identifier: ko-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="https://wandb.me/alerts-colab" >}}

run 이 중단되거나 사용자 정의 트리거가 발생할 때 Slack 또는 이메일로 알림을 받을 수 있습니다. 예를 들어, 트레이닝 루프의 그레이디언트가 급격히 커지거나(NaN 리포트) ML 파이프라인의 한 단계가 완료될 때 알림을 만들 수 있습니다. 알림은 개인 Project와 팀 Project 모두에서 run 을 초기화할 때 적용됩니다.

그리고 Slack(또는 이메일)에서 W&B Alerts 메시지를 확인할 수 있습니다:

{{< img src="/images/track/send_alerts_slack.png" alt="Slack alert setup" >}}

{{% alert %}}
W&B Alerts를 사용하려면 코드에 `run.alert()`를 추가해야 합니다. 코드를 수정하지 않고도, [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}})를 이용해 W&B 내 이벤트 발생 시 Slack에 알릴 수 있습니다. 예를 들어, [artifact]({{< relref path="/guides/core/artifacts" lang="ko" >}}) 버전이 생성될 때나 [run metric]({{< relref path="/guides/models/track/runs.md" lang="ko" >}}) 값이 임계값을 넘거나 변할 때 등입니다.

예를 들어, Automation은 새 버전이 생성될 때 Slack 채널에 알릴 수 있고, artifact에 `production` 에일리어스가 추가되면 자동으로 테스트 webhook을 실행하거나, run 의 `loss` 값이 허용 범위에 있을 때만 검증 작업을 시작할 수도 있습니다.

[Automations 개요]({{< relref path="/guides/core/automations/" lang="ko" >}})를 읽어보거나 [Automation 만들기]({{< relref path="/guides/core/automations/create-automations/" lang="ko" >}})를 참고하세요.
{{% /alert %}}


## 알림 만들기

{{% alert %}}
다음 가이드는 멀티 테넌트 클라우드 환경에서의 알림 설정에 대해 설명합니다.

[W&B Server]({{< relref path="/guides/hosting/" lang="ko" >}})를 Private Cloud 또는 W&B Dedicated Cloud에서 사용하는 경우, [W&B Server에서 Slack 알림 설정하기]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ko" >}}) 문서를 참고해 Slack 알림을 설정하세요.
{{% /alert %}}

알림을 설정하려면 아래 단계대로 진행하세요. 각 단계는 아래에서 자세히 설명합니다.

1. W&B [User Settings](https://wandb.ai/settings)에서 Alerts를 활성화합니다.
2. 코드에 `run.alert()`를 추가합니다.
3. 설정을 테스트합니다.

### 1. W&B User Settings에서 Alerts 활성화

[User Settings](https://wandb.ai/settings)에서 다음을 진행하세요:

* **Alerts** 섹션으로 이동합니다.
* `run.alert()`에서 알림을 받을 수 있도록 **Scriptable run alerts**를 켭니다.
* **Connect Slack**을 이용해 알림을 받을 Slack 채널을 선택합니다. 알림을 비공개로 받을 수 있는 **Slackbot** 채널 사용을 권장합니다.
* **Email** 알림은 W&B 가입 시 사용한 이메일로 전송됩니다. 다양한 알림이 메일함을 가득 채우지 않도록 이메일 필터를 만들어 따로 분리해 놓을 것을 권장합니다.

W&B Alerts를 처음 설정할 때 또는 알림 수신 방식을 변경하고 싶을 때만 이 과정을 한 번만 진행하면 됩니다.

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. 코드에 `run.alert()` 추가

원하는 위치(Notebook 또는 Python 스크립트)에 `run.alert()`를 삽입하세요.

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. 설정 테스트

Slack이나 이메일에서 알림 메시지를 확인하세요. 만약 알림을 받지 못했다면 [User Settings](https://wandb.ai/settings)에서 **Scriptable Alerts** 알림이 이메일 또는 Slack에서 활성화되어 있는지 확인하세요.

## 예시

아래 간단한 예시는 정확도(acc)가 임계값(threshold)보다 낮으면 경고 알림을 보냅니다. 이 예시에서는 최소 5분 간격으로만 알림을 보냅니다.

```python
import wandb
from wandb import AlertLevel

run = wandb.init()

if acc < threshold:
    run.alert(
        title="Low accuracy",
        text=f"Accuracy {acc} is below the acceptable threshold {threshold}",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## 사용자 태그 또는 멘션하기

`@` 뒤에 Slack 사용자 ID를 입력하면 본인 또는 동료를 알림의 제목이나 본문에서 태그할 수 있습니다. Slack 사용자 ID는 해당 Slack 프로필 페이지에서 확인할 수 있습니다.

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## 팀 단위 알림 설정

팀 관리자는 팀 설정 페이지(`wandb.ai/teams/your-team`)에서 팀 알림을 설정할 수 있습니다.

팀 알림은 팀 소속 모든 인원에게 적용됩니다. 알림을 비공개로 받으려면 **Slackbot** 채널 사용을 추천합니다.

## 알림이 전달될 Slack 채널 변경

알림을 보낼 채널을 변경하려면 **Disconnect Slack**을 클릭한 뒤 다시 연결하세요. 재연결 후 원하는 다른 Slack 채널을 선택하면 됩니다.