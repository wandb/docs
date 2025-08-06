---
title: ジョブやオートメーションでシークレットを指定できますか？例えば、ユーザーに直接見せたくない APIキー などです。
menu:
  launch:
    identifier: secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。以下の手順に従ってください。

1. 次のコマンドを使って、run 用に指定した namespace に Kubernetes シークレットを作成します。  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成した後、run 開始時にシークレットを注入できるようにキューを設定してください。シークレットはクラスター管理者のみが確認可能で、エンドユーザーには表示されません。