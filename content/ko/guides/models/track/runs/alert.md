---
title: Send an alert
description: Python 코 드 에서 트리거된 알림을 Slack 또는 이메일로 보내기
menu:
  default:
    identifier: ko-guides-models-track-runs-alert
    parent: what-are-runs
---

{{< cta-button colabLink="http://wandb.me/alerts-colab" >}}

run 이 충돌하거나 사용자 정의 트리거를 사용하는 경우 Slack 또는 이메일로 알림을 생성합니다. 예를 들어 트레이닝 루프의 그래디언트가 폭발하기 시작하거나 (NaN을 reports) ML 파이프라인의 단계가 완료되면 알림을 생성할 수 있습니다. 알림은 개인 및 팀 Projects를 포함하여 run 을 초기화하는 모든 프로젝트에 적용됩니다.

그런 다음 Slack (또는 이메일)에서 W&B Alerts 메시지를 확인하세요.

{{< img src="/images/track/send_alerts_slack.png" alt="" >}}

## 알림을 만드는 방법

{{% alert %}}
다음 가이드는 멀티 테넌트 클라우드의 알림에만 적용됩니다.

프라이빗 클라우드 또는 W&B 전용 클라우드에서 [W&B Server]({{< relref path="/guides/hosting/" lang="ko" >}})를 사용하는 경우 [이 문서]({{< relref path="/guides/hosting/monitoring-usage/slack-alerts.md" lang="ko" >}})를 참조하여 Slack 알림을 설정하세요.
{{% /alert %}}

알림을 설정하는 데는 두 가지 주요 단계가 있습니다.

1. W&B [사용자 설정](https://wandb.ai/settings)에서 Alerts를 켭니다.
2. 코드에 `run.alert()`를 추가합니다.
3. 알림이 올바르게 설정되었는지 확인합니다.

### 1. W&B 사용자 설정에서 Alerts를 켭니다.

[사용자 설정](https://wandb.ai/settings)에서:

* **Alerts** 섹션으로 스크롤합니다.
* `run.alert()`에서 알림을 받으려면 **Scriptable run alerts**를 켭니다.
* **Connect Slack**을 사용하여 알림을 게시할 Slack 채널을 선택합니다. 알림을 비공개로 유지하기 때문에 **Slackbot** 채널을 권장합니다.
* **Email**은 W&B에 가입할 때 사용한 이메일 어드레스로 전송됩니다. 이러한 모든 알림이 폴더로 이동하여 받은 편지함을 채우지 않도록 이메일에서 필터를 설정하는 것이 좋습니다.

W&B Alerts를 처음 설정하거나 알림을 받는 방법을 수정하려는 경우에만 이 작업을 수행하면 됩니다.

{{< img src="/images/track/demo_connect_slack.png" alt="Alerts settings in W&B User Settings" >}}

### 2. 코드에 `run.alert()`를 추가합니다.

알림을 트리거하려는 모든 위치에서 코드 (노트북 또는 Python 스크립트)에 `run.alert()`를 추가합니다.

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. Slack 또는 이메일 확인

Slack 또는 이메일에서 알림 메시지를 확인합니다. 아무것도 받지 못한 경우 [사용자 설정](https://wandb.ai/settings)에서 **Scriptable Alerts**에 대해 이메일 또는 Slack이 켜져 있는지 확인하세요.

### 예시

이 간단한 알림은 정확도가 임계값 아래로 떨어지면 경고를 보냅니다. 이 예에서는 최소 5분 간격으로 알림을 보냅니다.

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

## 사용자 태그 또는 멘션하는 방법

알림의 제목 또는 텍스트에서 자신이나 동료를 태그하려면 @ 기호 다음에 Slack 사용자 ID를 사용하세요. Slack 프로필 페이지에서 Slack 사용자 ID를 찾을 수 있습니다.

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## 팀 알림

팀 관리자는 팀 설정 페이지 `wandb.ai/teams/your-team`에서 팀에 대한 알림을 설정할 수 있습니다.

팀 알림은 팀의 모든 사용자에게 적용됩니다. W&B는 알림을 비공개로 유지하기 때문에 **Slackbot** 채널을 사용하는 것이 좋습니다.

## 알림을 보낼 Slack 채널 변경

알림을 보낼 채널을 변경하려면 **Disconnect Slack**을 클릭한 다음 다시 연결합니다. 다시 연결한 후 다른 Slack 채널을 선택합니다.
