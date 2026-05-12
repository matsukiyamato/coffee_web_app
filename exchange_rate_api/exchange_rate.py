from dotenv import load_dotenv
import os
import requests
import csv
from datetime import datetime


load_dotenv()
API_KEY = os.getenv("API_KEY")

url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"
res = requests.get(url)
data = res.json()

#ドル→円
usd_jpy = data["conversion_rates"]["JPY"]

print("USD/JPY =", usd_jpy)

# 現在日時
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# CSV保存
with open("exchange_rate.csv", "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # 1行書き込み
    writer.writerow([now, usd_jpy])

print("CSVに保存しました")