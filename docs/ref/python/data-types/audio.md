# Audio



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1027-L1176)



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



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1134-L1136)

```python
@classmethod
durations(
 audio_list
)
```




### `path_is_reference`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1072-L1074)

```python
@classmethod
path_is_reference(
 path
)
```




### `resolve_ref`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1150-L1162)

```python
resolve_ref()
```




### `sample_rates`



[View source](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/data_types.py#L1138-L1140)

```python
@classmethod
sample_rates(
 audio_list
)
```






