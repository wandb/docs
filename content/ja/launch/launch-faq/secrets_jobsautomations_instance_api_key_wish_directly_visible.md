---
title: jobs/automations 用にシークレットを指定できますか？ たとえば、ユーザーに直接表示したくない API キーなどは？
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。次の手順に従ってください:

1. 次のコマンドを使用して、runs 用に指定した Namespace に Kubernetes の Secret を作成します:  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. Secret を作成したら、runs の開始時にその Secret を注入するようにキューを設定します。Secret を閲覧できるのは クラスター 管理者のみで、エンド ユーザーには表示されません。