---
title: 알림 및 알림 설정하기
description: 새로운 모델 버전이 모델 레지스트리에 연결되면 Slack 알림을 받아보세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

모델 레지스트리에 새로운 model version 이 연결될 때 Slack 알림을 받을 수 있습니다.

1. [W&B Model Registry 앱](https://wandb.ai/registry/model)으로 이동합니다.
2. 알림을 받고 싶은 registered model 을 선택합니다.
3. **Connect Slack** 버튼을 클릭합니다.
    {{< img src="/images/models/connect_to_slack.png" alt="Connect to Slack" >}}
4. OAuth 페이지에 나타나는 안내에 따라 Slack workspace 에서 W&B 연동을 설정합니다.

팀을 위해 Slack 알림을 설정하면, 각 registered model 별로 알림을 받을지 선택할 수 있습니다.

{{% alert %}}
Slack 알림이 팀에 대해 이미 설정되어 있는 경우, **Connect Slack** 버튼 대신 **New model version linked to...** 토글이 나타납니다.
{{% /alert %}}

아래 스크린샷은 Slack 알림이 설정된 FMNIST classifier registered model 의 예시입니다.

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="Slack notification example" >}}

새로운 model version 이 FMNIST classifier registered model 에 연결될 때마다, 연결된 Slack 채널에 메시지가 자동으로 전송됩니다.