---
description: Get Slack notifications when a new model version is linked to the model
  registry.
displayed_sidebar: default
---

# 알림 및 알림 생성하기

새로운 모델 버전이 모델 레지스트리에 연결될 때 Slack 알림을 받습니다.

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 W&B 모델 레지스트리 앱으로 이동합니다.
2. 알림을 받고 싶은 등록된 모델을 선택합니다.
3. **Connect Slack** 버튼을 클릭합니다.
    ![](/images/models/connect_to_slack.png)
4. OAuth 페이지에 나타나는 지침을 따라 Slack 워크스페이스에서 W&B를 활성화합니다.

팀을 위한 Slack 알림을 설정하면, 알림을 받고 싶은 등록된 모델을 선택할 수 있습니다.

:::info
팀을 위한 Slack 알림이 구성되어 있으면, **Connect Slack** 버튼 대신 **New model version linked to...**라고 표시되는 토글이 나타납니다.
:::

아래 스크린샷은 Slack 알림이 설정된 FMNIST 분류기 등록 모델을 보여줍니다.

![](/images/models/conect_to_slack_fmnist.png)

새로운 모델 버전이 FMNIST 분류기 등록 모델에 연결될 때마다 연결된 Slack 채널에 자동으로 메시지가 게시됩니다.