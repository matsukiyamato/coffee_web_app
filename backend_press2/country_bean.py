from flask import Flask, request
import csv

app = Flask(__name__)

#POSTで /save に送られてきたデータを受け取る
@app.route("/save", methods=["POST"])
def save():

    #原産地名のname属性:country　豆の種類のname属性:bean
    country = request.form["country"]
    bean = request.form["bean"]

    print(country)
    print(bean)

    with open("coffee_data.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([country, bean])

    return "保存完了"

if __name__ == "__main__":
    app.run(debug=True)