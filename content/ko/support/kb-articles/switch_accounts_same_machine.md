---
title: 같은 컴퓨터에서 계정을 전환하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-switch_accounts_same_machine
support:
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

두 개의 W&B 계정을 같은 컴퓨터에서 관리하려면, 두 API 키를 파일에 저장하세요. 아래 코드를 저장소에 사용하여 키를 안전하게 전환할 수 있으며, 비밀 키가 소스 컨트롤에 포함되는 것을 방지할 수 있습니다.

```python
# 만약 ~/keys.json 파일이 존재하면,
if os.path.exists("~/keys.json"):
    # 환경 변수에 work_account의 API 키를 설정합니다.
    os.environ["WANDB_API_KEY"] = json.loads("~/keys.json")["work_account"]
```
