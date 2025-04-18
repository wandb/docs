---
title: How do I switch between accounts on the same machine?
menu:
  support:
    identifier: ko-support-kb-articles-switch_accounts_same_machine
support:
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

동일한 장비에서 두 개의 W&B 계정을 관리하려면 두 개의 API 키를 파일에 저장하세요. 비밀 키가 소스 제어에 체크인되지 않도록 다음 코드를 저장소에서 사용하여 키를 안전하게 전환하세요.

```python
if os.path.exists("~/keys.json"):
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
