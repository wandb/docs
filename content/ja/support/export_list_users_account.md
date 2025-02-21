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

W&B の組織からユーザーのリストをエクスポートするには、管理者が次のコードを使用して SCIM API を利用します。

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # 組織の管理者のユーザー名
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

スクリプトを修正して、必要に応じて出力を保存します。