---
title: How do I export a list of users from my W&B Organisation?
menu:
  support:
    identifier: ko-support-kb-articles-export_list_users_account
support:
- administrator
- user management
toc_hide: true
type: docs
url: /support/:filename
---

W&B 조직에서 사용자 목록을 내보내려면 관리자는 다음 코드를 사용하여 SCIM API를 사용합니다.

```python
import base64
import requests

def encode_base64(username, key):
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # Organization admin username
key = ''  # API 키
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

필요에 따라 출력을 저장하도록 스크립트를 수정하세요.
