---
title: W&B Sweeps를 AWS Batch, ECS 등과 같은 클라우드 인프라에서 사용할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-sweeps_cloud_infrastructures_such_aws_batch_ecs
support:
- 스윕
- 'aws

  '
toc_hide: true
type: docs
url: /support/:filename
---

`sweep_id` 를 공개하여 모든 W&B 스윕 에이전트가 엑세스할 수 있도록 하려면, 이 에이전트들이 `sweep_id` 를 읽고 실행할 수 있는 메소드를 구현하세요.

예를 들어, Amazon EC2 인스턴스를 실행하고 그 위에서 `wandb agent` 를 실행할 수 있습니다. SQS 큐를 사용해 여러 EC2 인스턴스에 `sweep_id` 를 브로드캐스트할 수 있습니다. 각 인스턴스는 큐에서 `sweep_id` 를 가져와 프로세스를 시작할 수 있습니다.