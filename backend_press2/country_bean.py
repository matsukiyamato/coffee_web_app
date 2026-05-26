from flask import Flask, request, render_template
from exchange_rate_api.exchange_rate import get_exchange_rate
from datetime import datetime
import csv
import pandas as pd


app = Flask(
    __name__,
    template_folder="../templates"
)



@app.route("/")
def index():
    return render_template("index.html")

#POSTで /save に送られてきたデータを受け取る
@app.route("/save", methods=["POST"])
def save():


    #原産地名:country　豆の種類:bean
    country = request.form["country"]
    bean = request.form["bean"]


    
    with open("coffee_data.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([country, bean])

    if country == "ブラジル":

        #API取得後datetimeで時刻を表示する

        today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ex_rate = get_exchange_rate()

        # グラフ用CSV読み込み

        # 降水量
        rain_df = pd.read_csv("CSV/rainfall.csv")

        # 予想価格
        price_df = pd.read_csv("CSV/predicted_price.csv")

        # 為替
        exchange_df = pd.read_csv("CSV/exchange_rate.csv")

        # 生産量
        production_df = pd.read_csv("CSV/production_volume.csv")


        # date列を日付型に変換
        rain_df["date"] = pd.to_datetime(rain_df["date"])

        price_df["date"] = pd.to_datetime(price_df["date"])

        exchange_df["date"] = pd.to_datetime(exchange_df["date"])

        production_df["date"] = pd.to_datetime(production_df["date"])


        # 日付をずらす

        # 降水量 → 1年前のデータを使う
        rain_df["date"] = rain_df["date"] + pd.DateOffset(years=1)

        # 為替 → 1週間前のデータを使う
        exchange_df["date"] = exchange_df["date"] + pd.DateOffset(weeks=1)

        # 生産量 → 1か月前のデータを使う
        production_df["date"] = production_df["date"] + pd.DateOffset(months=1)


        # 月ごとの平均に変換
        monthly_rain = rain_df.resample(
            "ME",
            on="date"
        )["rainfall_mm"].mean()

        monthly_price = price_df.resample(
            "ME",
            on="date"
        )["predicted_price"].mean()

        monthly_exchange = exchange_df.resample(
            "ME",
            on="date"
        )["usd_jpy"].mean()

        monthly_production = production_df.resample(
            "ME",
            on="date"
        )["production_ton"].mean()


        # グラフ用データ
        labels = monthly_price.index.strftime("%Y-%m").tolist()

        rain_data = monthly_rain.tolist()

        price_data = monthly_price.tolist()

        #exchange_data = monthly_exchange.tolist()

        #production_data = monthly_production.tolist()





        return render_template(
            "result.html",
            ex_rate=ex_rate,
            today_date=today_date,

            labels=labels,
            rain_data=rain_data,
            price_data=price_data
        )

        
   
    

if __name__ == "__main__":
    app.run(debug=True)