---
title: Docker キュー内の複数のジョブが同じ Artifact をダウンロードする場合、キャッシュは利用されますか？それとも毎回 Run ごとに再ダウンロードされますか？
menu:
  launch:
    identifier: docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュは存在しません。各 Launch ジョブは独立して動作します。キューの設定で Docker の引数を使って、キューやエージェントが共有キャッシュをマウントするように設定してください。

さらに、特定のユースケースでは W&B Artifacts キャッシュを永続ボリュームとしてマウントすることもできます。