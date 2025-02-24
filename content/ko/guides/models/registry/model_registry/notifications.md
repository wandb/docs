---
title: Create alerts and notifications
description: 새로운 모델 버전이 모델 레지스트리에 연결될 때 Slack 알림을 받으세요.
menu:
  default:
    identifier: ko-guides-models-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

새로운 모델 버전이 모델 레지스트리에 연결될 때 Slack 알림을 받으세요.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)의 W&B Model Registry 앱으로 이동합니다.
2. 알림을 받고 싶은 registered model을 선택합니다.
3. **Connect Slack** 버튼을 클릭합니다.
    {{< img src="/images/models/connect_to_slack.png" alt="" >}}
4. OAuth 페이지에 나타나는 지침에 따라 Slack workspace에서 W&B를 활성화합니다.

팀에 대한 Slack 알림을 구성한 후 알림을 받을 registered model을 선택할 수 있습니다.

{{% alert %}}
팀에 대해 Slack 알림을 구성한 경우 **Connect Slack** 버튼 대신 **New model version linked to...** 토글이 나타납니다.
{{% /alert %}}

아래 스크린샷은 Slack 알림이 있는 FMNIST classifier registered model을 보여줍니다.

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="" >}}

새로운 모델 버전이 FMNIST classifier registered model에 연결될 때마다 연결된 Slack 채널에 메시지가 자동으로 게시됩니다.