---
title: 浮動小数点数
menu:
  reference:
    identifier: ja-ref-query-panel-float
---

## チェーン可能な演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>
2 つの値が等しくないかを判定します。
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |
#### 戻り値
2 つの値が等しくないかどうか。
<h3 id="number-modulo"><code>number-modulo</code></h3>
ある [number](number.md) を別の [number](number.md) で割り、余りを返します
| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る側の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の剰余
<h3 id="number-mult"><code>number-mult</code></h3>
2 つの [number](number.md) を掛けます
| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [number](number.md) |
| `rhs` | 2 つ目の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の積
<h3 id="number-powBinary"><code>number-powBinary</code></h3>
[number](number.md) を指数でべき乗します
| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |
#### 戻り値
底の [number](number.md) を n 乗した値
<h3 id="number-add"><code>number-add</code></h3>
2 つの [number](number.md) を加算します
| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [number](number.md) |
| `rhs` | 2 つ目の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の和
<h3 id="number-sub"><code>number-sub</code></h3>
ある [number](number.md) から別の [number](number.md) を減算します
| 引数 |  |
| :--- | :--- |
| `lhs` | 減算対象の [number](number.md) |
| `rhs` | 減算する [number](number.md) |
#### 戻り値
2 つの [number](number.md) の差
<h3 id="number-div"><code>number-div</code></h3>
ある [number](number.md) を別の [number](number.md) で割ります
| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る側の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の商
<h3 id="number-less"><code>number-less</code></h3>
ある [number](number.md) が別のものより小さいかを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目より小さいかどうか
<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>
ある [number](number.md) が別のもの以下かを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目以下かどうか
<h3 id="number-equal"><code>number-equal</code></h3>
2 つの値が等しいかを判定します。
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |
#### 戻り値
2 つの値が等しいかどうか。
<h3 id="number-greater"><code>number-greater</code></h3>
ある [number](number.md) が別のものより大きいかを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目より大きいかどうか
<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>
ある [number](number.md) が別のもの以上かを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目以上かどうか
<h3 id="number-negate"><code>number-negate</code></h3>
[number](number.md) の符号を反転します
| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数値 |
#### 戻り値
[number](number.md)
<h3 id="number-toString"><code>number-toString</code></h3>
[number](number.md) を文字列に変換します
| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |
#### 戻り値
その [number](number.md) の文字列表現
<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>
[number](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒、31536000000000 未満の値はミリ秒、31536000000000000 未満の値はマイクロ秒、31536000000000000000 未満の値はナノ秒に変換されます。
| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |
#### 戻り値
タイムスタンプ
<h3 id="number-abs"><code>number-abs</code></h3>
[number](number.md) の絶対値を計算します
| 引数 |  |
| :--- | :--- |
| `n` | [number](number.md) |
#### 戻り値
その [number](number.md) の絶対値
## リスト演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>
2 つの値が等しくないかを判定します。
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |
#### 戻り値
2 つの値が等しくないかどうか。
<h3 id="number-modulo"><code>number-modulo</code></h3>
ある [number](number.md) を別の [number](number.md) で割り、余りを返します
| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る側の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の剰余
<h3 id="number-mult"><code>number-mult</code></h3>
2 つの [number](number.md) を掛けます
| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [number](number.md) |
| `rhs` | 2 つ目の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の積
<h3 id="number-powBinary"><code>number-powBinary</code></h3>
[number](number.md) を指数でべき乗します
| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |
#### 戻り値
底の [number](number.md) を n 乗した値
<h3 id="number-add"><code>number-add</code></h3>
2 つの [number](number.md) を加算します
| 引数 |  |
| :--- | :--- |
| `lhs` | 1 つ目の [number](number.md) |
| `rhs` | 2 つ目の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の和
<h3 id="number-sub"><code>number-sub</code></h3>
ある [number](number.md) から別の [number](number.md) を減算します
| 引数 |  |
| :--- | :--- |
| `lhs` | 減算対象の [number](number.md) |
| `rhs` | 減算する [number](number.md) |
#### 戻り値
2 つの [number](number.md) の差
<h3 id="number-div"><code>number-div</code></h3>
ある [number](number.md) を別の [number](number.md) で割ります
| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る側の [number](number.md) |
#### 戻り値
2 つの [number](number.md) の商
<h3 id="number-less"><code>number-less</code></h3>
ある [number](number.md) が別のものより小さいかを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目より小さいかどうか
<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>
ある [number](number.md) が別のもの以下かを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目以下かどうか
<h3 id="number-equal"><code>number-equal</code></h3>
2 つの値が等しいかを判定します。
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する 1 つ目の値。 |
| `rhs` | 比較する 2 つ目の値。 |
#### 戻り値
2 つの値が等しいかどうか。
<h3 id="number-greater"><code>number-greater</code></h3>
ある [number](number.md) が別のものより大きいかを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目より大きいかどうか
<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>
ある [number](number.md) が別のもの以上かを確認します
| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |
#### 戻り値
最初の [number](number.md) が 2 つ目以上かどうか
<h3 id="number-negate"><code>number-negate</code></h3>
[number](number.md) の符号を反転します
| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数値 |
#### 戻り値
[number](number.md)
<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>
最大の [number](number.md) のインデックスを求めます
| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) のインデックスを求める対象の [numbers](number.md) の _リスト_ |
#### 戻り値
最大の [number](number.md) のインデックス
<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>
最小の [number](number.md) のインデックスを求めます
| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) のインデックスを求める対象の [numbers](number.md) の _リスト_ |
#### 戻り値
最小の [number](number.md) のインデックス
<h3 id="numbers-avg"><code>numbers-avg</code></h3>
[numbers](number.md) の平均
| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を計算する [numbers](number.md) の _リスト_ |
#### 戻り値
[numbers](number.md) の平均
<h3 id="numbers-max"><code>numbers-max</code></h3>
最大値
| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) を求める対象の [numbers](number.md) の _リスト_ |
#### 戻り値
最大の [number](number.md)
<h3 id="numbers-min"><code>numbers-min</code></h3>
最小値
| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) を求める対象の [numbers](number.md) の _リスト_ |
#### 戻り値
最小の [number](number.md)
<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>
[numbers](number.md) の標準偏差
| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [numbers](number.md) の _リスト_ |
#### 戻り値
[numbers](number.md) の標準偏差
<h3 id="numbers-sum"><code>numbers-sum</code></h3>
[numbers](number.md) の合計
| 引数 |  |
| :--- | :--- |
| `numbers` | 合計する [numbers](number.md) の _リスト_ |
#### 戻り値
[numbers](number.md) の合計
<h3 id="number-toString"><code>number-toString</code></h3>
[number](number.md) を文字列に変換します
| 引数 |  |
| :--- | :--- |
| `in` | 変換する数値 |
#### 戻り値
その [number](number.md) の文字列表現
<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>
[number](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒、31536000000000 未満の値はミリ秒、31536000000000000 未満の値はマイクロ秒、31536000000000000000 未満の値はナノ秒に変換されます。
| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数値 |
#### 戻り値
タイムスタンプ
<h3 id="number-abs"><code>number-abs</code></h3>
[number](number.md) の絶対値を計算します
| 引数 |  |
| :--- | :--- |
| `n` | [number](number.md) |
#### 戻り値
その [number](number.md) の絶対値