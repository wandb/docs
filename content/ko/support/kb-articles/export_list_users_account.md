---
title: W&B 조직에서 사용자 목록을 내보내려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-export_list_users_account
support:
- 관리자
- 사용자 관리
toc_hide: true
type: docs
url: /support/:filename
---

W&B 조직에서 사용자 목록을 내보내려면, 관리자가 아래 코드와 같이 SCIM API 를 사용합니다.

```python
import base64
import requests

def encode_base64(username, key):
    # 인증 문자열을 base64로 인코딩합니다.
    auth_string = f'{username}:{key}'
    return base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

username = ''  # 조직 관리자 사용자 이름
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
    # 각 사용자의 userName 과 이메일 값을 users 리스트에 추가합니다.
    users.append([user['userName'], user['emails']['Value']])
```

출력 결과를 원하는 형태로 저장하도록 스크립트를 수정하세요.