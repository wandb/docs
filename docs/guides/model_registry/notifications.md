---
title: Create alerts and notifications
description: 새로운 모델 버전이 모델 레지스트리와 연결될 때 Slack 알림을 받으세요.
displayed_sidebar: default
---

새 모델 버전이 모델 레지스트리에 연결될 때 Slack 알림을 받으세요.

1. W&B Model Registry 앱으로 이동하세요: [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. 알림을 받으려는 등록된 모델을 선택하세요.
3. **Connect Slack** 버튼을 클릭하세요.
   ![](/images/models/connect_to_slack.png)
4. OAuth 페이지에 나타나는 안내에 따라 Slack 워크스페이스에서 W&B를 활성화하세요.

팀에 대한 Slack 알림을 설정한 후, 등록된 모델을 선택하여 알림을 받을 수 있습니다.

:::info
팀에 대한 Slack 알림이 설정되어 있으면 **Connect Slack** 버튼 대신 **New model version linked to...** 라고 읽히는 토글이 나타납니다.
:::

아래 스크린샷은 Slack 알림이 있는 FMNIST 분류기 등록 모델을 보여줍니다.

![](/images/models/conect_to_slack_fmnist.png)

FMNIST 분류기 등록 모델에 새 모델 버전이 연결될 때마다 연결된 Slack 채널에 메시지가 자동으로 게시됩니다.