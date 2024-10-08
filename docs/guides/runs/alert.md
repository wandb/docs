---
title: Send an alert
description: Python 코드에서 트리거된 알림을 Slack 또는 이메일로 전송하세요
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="http://wandb.me/alerts-colab"/>

run이 중지되거나 사용자 정의 트리거가 발생할 경우 Slack 또는 이메일로 알림을 생성하세요. 예를 들어, 트레이닝 루프의 그레이디언트가 급격히 증가하기 시작하거나(리포트 NaN) ML 파이프라인의 단계가 완료될 때 알림을 생성할 수 있습니다. 알림은 개인 프로젝트와 팀 프로젝트를 포함하여 run을 초기화하는 모든 프로젝트에 적용됩니다.

그리고 Slack (또는 이메일)에서 W&B Alerts 메시지를 확인하세요:

![](/images/track/send_alerts_slack.png)

## 알림을 생성하는 방법

:::info 안내
다음 가이드는 멀티 테넌트 클라우드의 알림에만 적용됩니다.

[W&B Server](../hosting/intro.md)를 프라이빗 클라우드나 W&B 전용 클라우드에서 사용 중인 경우, Slack 알림을 설정하려면 [이 문서](../hosting/monitoring-usage/slack-alerts.md)를 참조하세요.
:::

알림을 설정하는 두 가지 주요 단계가 있습니다:

1. W&B [User Settings](https://wandb.ai/settings)에서 알림 켜기
2. 코드에 `run.alert()` 추가하기
3. 알림이 제대로 설정되었는지 확인하기

### 1. W&B User Settings에서 알림 켜기

[User Settings](https://wandb.ai/settings)에서:

* **Alerts** 섹션으로 스크롤하세요
* `run.alert()`로부터 알림을 받으려면 **Scriptable run alerts**를 켜세요
* **Connect Slack**을 사용하여 알림을 게시할 Slack 채널을 선택하세요. 알림을 비공개로 유지하기 위해 **Slackbot** 채널을 추천합니다.
* **Email**은 W&B에 가입했을 때 사용한 이메일 주소로 보내집니다. 이러한 알림이 폴더로 이동하고 받은 편지함을 가득 채우지 않도록 하려면 이메일에서 필터를 설정하는 것을 추천합니다.

W&B Alerts를 처음 설정할 때 또는 알림 수신 방식을 변경하고 싶을 때만 이 작업을 수행해야 합니다.

![W&B User Settings의 Alerts 설정](/images/track/demo_connect_slack.png)

### 2. 코드에 `run.alert()` 추가하기

원하는 위치에 `run.alert()`를 코드(노트북 또는 Python 스크립트)로 추가하세요.

```python
import wandb

run = wandb.init()
run.alert(title="High Loss", text="Loss is increasing rapidly")
```

### 3. Slack 또는 이메일 확인하기

Slack 또는 이메일에서 알림 메시지를 확인하세요. 아무것도 수신되지 않았다면, [User Settings](https://wandb.ai/settings)에서 **Scriptable Alerts**에 대해 이메일 또는 Slack이 켜져 있는지 확인하세요.

### 예제

이 간단한 알림은 정확도가 임계값 이하로 떨어질 때 경고를 보냅니다. 이 예제에서는 최소 5분 간격으로만 알림을 보냅니다.

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

## 사용자 태그하거나 언급하는 방법

at 기호 `@`에 Slack 사용자 ID를 붙여 알림의 제목 또는 텍스트에서 자신이나 동료를 태그하세요. Slack 프로필 페이지에서 Slack 사용자 ID를 찾을 수 있습니다.

```python
run.alert(title="Loss is NaN", text=f"Hey <@U1234ABCD> loss has gone to NaN")
```

## 팀 알림

팀 관리자는 팀 설정 페이지에서 팀 알림을 설정할 수 있습니다: `wandb.ai/teams/your-team`.

팀 알림은 팀의 모든 사람에게 적용됩니다. W&B는 알림을 비공개로 유지하기 위해 **Slackbot** 채널을 사용하는 것을 권장합니다.

## 알림을 전송할 Slack 채널 변경하기

알림을 전송할 채널을 변경하려면 **Disconnect Slack**을 클릭한 다음 다시 연결하세요. 재연결 후 다른 Slack 채널을 선택하세요.

## FAQ(s)

### "Run Finished" 알림은 노트북에서 작동합니까?

아니요. **Run Finished** 알림(User Settings에서 **Run Finished** 설정으로 켜짐)은 Python 스크립트에서만 작동하며 Jupyter 노트북 환경에서는 각 셀 실행마다 알림이 발생하지 않도록 비활성화됩니다.

대신 노트북 환경에서는 `wandb.alert()`을 사용하세요.