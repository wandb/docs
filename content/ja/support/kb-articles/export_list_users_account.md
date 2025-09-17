---
title: 自分の W&B 組織からユーザーの一覧をエクスポートするには？
menu:
  support:
    identifier: ja-support-kb-articles-export_list_users_account
support:
- 管理者
- ユーザー管理
toc_hide: true
type: docs
url: /support/:filename
---

W&B の組織から Users の一覧をエクスポートするには、管理者は次のコードのように SCIM API を使用します:

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # 組織管理者のユーザー名
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

必要に応じて出力を保存するようにスクリプトを修正してください。