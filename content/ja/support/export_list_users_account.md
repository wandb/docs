---
title: How do I export a list of users from my W&B Organisation?
menu:
  support:
    identifier: ja-support-export_list_users_account
tags:
- administrator
- user management
toc_hide: true
type: docs
---

W&B の Organization から ユーザー のリストをエクスポートするには、管理者 は次の コード で SCIM API を使用します。

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # Organization 管理者 ユーザー名
key = ''  # APIキー
scim_base_url = 'https://api.wandb.ai/scim/v2'
users_endpoint = f'{scim_base_url}/Users'
headers = {
    'Authorization': f'Basic {encode_base64(username, key)}',
    'Content-Type': 'application/scim+json'
}

response = requests.get(users_endpoint, headers=headers)
users = []
for user in response.json()['Resources']:
    users.append([user['userName'], user['emails']['Value']])
```

必要に応じて、出力を保存するように スクリプト を変更します。
