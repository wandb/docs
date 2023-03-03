# Audio



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/data_types.py#L1026-L1171)



Wandb class for audio clips.

```python
Audio(
 data_or_path, sample_rate=None, caption=None
)
```





| Arguments | |
| :--- | :--- |
| `data_or_path` | (string or numpy array) A path to an audio file or a numpy array of audio data. |
| `sample_rate` | (int) Sample rate, required when passing in raw numpy array of audio data. |
| `caption` | (string) Caption to display with audio. |



## Methods

### `durations`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/data_types.py#L1129-L1131)

```python
@classmethod
durations(
 audio_list
)
```




### `resolve_ref`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/data_types.py#L1145-L1157)

```python
resolve_ref()
```




### `sample_rates`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/data_types.py#L1133-L1135)

```python
@classmethod
sample_rates(
 audio_list
)
```






