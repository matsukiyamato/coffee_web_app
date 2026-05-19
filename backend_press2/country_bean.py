from flask import Flask, request, render_template
from exchange_rate_api.exchange_rate import get_exchange_rate
from datetime import datetime
import csv


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

        #為替:ex_rate 取得日時:today_date
        return render_template(
            "result.html",
            ex_rate=ex_rate,
            today_date=today_date
        )

        
   
    

if __name__ == "__main__":
    app.run(debug=True)