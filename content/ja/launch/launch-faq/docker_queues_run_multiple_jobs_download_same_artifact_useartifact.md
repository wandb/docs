---
title: Dockerキュー内の複数のジョブが同じアーティファクトをダウンロードする場合、キャッシュは使用されますか、それとも毎回のrunで再ダウンロードされますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュは存在しません。各ローンチジョブは独立して動作します。キューの設定で Docker の引数を使用して、共有キャッシュをマウントするようにキューまたはエージェントを設定してください。

さらに、特定のユースケースに対して、W&B アーティファクトキャッシュを永続ボリュームとしてマウントします。