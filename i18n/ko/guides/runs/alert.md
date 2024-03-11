---
description: Send alerts, triggered from your Python code, to your Slack or email
displayed_sidebar: default
---

# wandb.alert로 알림 보내기

<head>
  <title>Python 코드에서 알림 보내기</title>
</head>

[**여기에서 Colab 노트북으로 시도해 보세요 →**](http://wandb.me/alerts-colab)

W&B Alerts를 사용하면 W&B Run이 중단되었거나 손실이 NaN으로 가거나 ML 파이프라인의 단계가 완료되는 등 사용자 정의 트리거에 도달했을 때 Slack이나 이메일을 통해 알림을 받을 수 있습니다. W&B Alerts는 개인 및 팀 프로젝트를 포함하여 run을 시작하는 모든 프로젝트에 적용됩니다.

알림을 이렇게 설정할 수 있습니다:

```python
text = f"정확도 {acc}가 허용 가능한 임계값 {thresh} 이하입니다"

wandb.alert(title="낮은 정확도", text=text)
```

그런 다음 Slack(또는 이메일)에서 W&B Alerts 메시지를 볼 수 있습니다:

![](/images/track/send_alerts_slack.png)

## 시작하기

:::info
다음 단계는 공용 클라우드에서 알림을 켜는 방법에만 해당됩니다.

프라이빗 클라우드 또는 전용 클라우드에서 [W&B 서버](../hosting/intro.md)를 사용하는 경우 Slack 알림을 설정하려면 [이 문서](../hosting/slack-alerts.md)를 참조하세요.
:::

코드에서 Slack 또는 이메일 알림을 보내려면 처음에 다음 2단계를 따르세요:

1. W&B [사용자 설정](https://wandb.ai/settings)에서 알림 켜기
2. 코드에 `wandb.alert()` 추가하기

### 1. W&B 사용자 설정에서 알림 켜기

[사용자 설정](https://wandb.ai/settings)에서:

* **알림** 섹션으로 스크롤
* `wandb.alert()`에서 알림을 받으려면 **스크립트 가능한 실행 알림** 켜기
* **Slack 연결**을 사용하여 알림을 게시할 Slack 채널 선택. 알림이 비공개로 유지되므로 **Slackbot** 채널을 권장합니다.
* **이메일**은 W&B에 가입할 때 사용한 이메일 주소로 전송됩니다. 이메일에서 이러한 알림이 모두 폴더로 이동하도록 필터를 설정하여 받은 편지함이 가득 차지 않도록 하는 것이 좋습니다.

알림을 설정하는 것은 처음이나 알림 수신 방식을 변경하고 싶을 때만 해야 합니다.

![W&B 사용자 설정의 알림 설정](/images/track/demo_connect_slack.png)

### 2. 코드에 \`wandb.alert()\` 추가하기

`wandb.alert()`를 코드(노트북 또는 Python 스크립트)에 트리거하고 싶은 곳에 추가하세요

```python
wandb.alert(title="높은 손실", text="손실이 급격히 증가하고 있습니다")
```

#### Slack이나 이메일 확인하기

Slack이나 이메일에서 알림 메시지를 확인하세요. 받지 못했다면 [사용자 설정](https://wandb.ai/settings)에서 **스크립트 가능한 알림**에 대해 이메일이나 Slack이 켜져 있는지 확인하세요.

## \`wandb.alert()\` 사용하기

| 인수                      | 설명                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title` (문자열)           | 알림의 짧은 설명, 예: "낮은 정확도"                                                                                          |
| `text` (문자열)            | 알림을 트리거한 사건에 대한 더 길고 자세한 설명                                                                             |
| `level` (선택사항)         | 알림의 중요도 — `AlertLevel.INFO`, `AlertLevel.WARN`, 또는 `AlertLevel.ERROR` 중 하나여야 합니다. `wandb`에서 `AlertLevel.xxx`을 가져올 수 있습니다 |
|                            |                                                                                                                                                       |
| `wait_duration` (선택사항) | 동일한 **title**으로 다른 알림을 보내기 전에 기다릴 초 수입니다. 이는 알림 스팸을 줄이는 데 도움이 됩니다                                           |

### 예시

이 간단한 알림은 정확도가 임계값 아래로 떨어질 때 경고를 보냅니다. 이 예에서는 적어도 5분 간격으로 알림을 보냅니다.

[코드 실행하기 →](http://wandb.me/alerts)

```python
import wandb
from wandb import AlertLevel

if acc < threshold:
    wandb.alert(
        title="낮은 정확도",
        text=f"정확도 {acc}가 허용 가능한 임계값 {threshold} 이하입니다",
        level=AlertLevel.WARN,
        wait_duration=300,
    )
```

## 추가 정보

### 태깅 / 사용자 언급하기

Slack에서 알림을 보낼 때 알림의 제목이나 텍스트에 `<@USER_ID>`를 추가하여 자신이나 동료를 @할 수 있습니다. Slack 사용자 ID는 해당 Slack 프로필 페이지에서 찾을 수 있습니다.

```python
wandb.alert(title="손실이 NaN입니다", text=f"안녕하세요 <@U1234ABCD>, 손실이 NaN으로 변했습니다")
```

### W&B 팀 알림

팀 관리자는 팀 설정 페이지 wandb.ai/teams/`your-team`에서 팀을 위한 알림을 설정할 수 있습니다. 이러한 알림은 팀의 모든 사람에게 적용됩니다. 알림이 비공개로 유지되므로 **Slackbot** 채널을 권장합니다.

### Slack 채널 변경하기

게시할 채널을 변경하려면 **Slack 연결 해제**를 클릭한 다음 다른 대상 채널을 선택하고 다시 연결하세요.

## FAQ

#### "Run Finished" 알림이 Jupyter 노트북에서 작동하나요?

**"Run Finished"** 알림(사용자 설정의 **"Run Finished"** 설정으로 켜짐)은 Python 스크립트에서만 작동하며, 셀 실행마다 알림 알림을 방지하기 위해 Jupyter 노트북 환경에서는 비활성화됩니다. 대신 Jupyter 노트북 환경에서 `wandb.alert()`를 사용하세요.

#### [W&B 서버](../hosting/intro.md)에서 알림을 활성화하는 방법은?