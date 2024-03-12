
# 마무리

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/wandb_run.py#L4093-L4104' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


run을 완료된 상태로 표시하고 모든 데이터의 업로드를 완료합니다.

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

이것은 동일한 프로세스에서 여러 run을 생성할 때 사용됩니다.
스크립트가 종료될 때 이 메소드를 자동으로 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  0 이외의 것으로 설정하여 run을 실패로 표시 |
|  `quiet` |  로그 출력을 최소화하려면 true로 설정 |