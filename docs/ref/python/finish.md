# finish

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/wandb_run.py#L4366-L4377' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

run을 완료로 표시하고, 모든 데이터를 업로드 완료합니다.

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

이 메소드는 같은 프로세스에서 여러 runs를 생성할 때 사용됩니다.
스크립트가 종료될 때 자동으로 이 메소드를 호출합니다.

| 인수 |  |
| :--- | :--- |
|  `exit_code` |  다른 값이 0으로 설정되면 run이 실패한 것으로 표시됩니다 |
|  `quiet` |  로그 출력을 최소화하려면 true로 설정합니다 |