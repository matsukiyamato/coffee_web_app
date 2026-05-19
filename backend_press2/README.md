# backend_press2

## 現在までの処理

フロント
 →
選択
 →
Flask受け取り
 →
CSV保存

---

## ファイル

backend_press2/country_bean.py

---

## 受け取るデータ

### 変数名

- 原産地名
  - name属性: `country`

- 豆の種類
  - name属性: `bean`

---

## 処理内容

POSTで `/save` に送られてきたデータを受け取る

```python
country = request.form["country"]
bean = request.form["bean"]