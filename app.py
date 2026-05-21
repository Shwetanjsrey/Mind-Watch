from flask import Flask, render_template, request
from src.risk_engine import analyze_risk

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    analysis = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()

        if input_text:
            analysis = analyze_risk(input_text)

    return render_template(
        "index.html",
        analysis=analysis,
        input_text=input_text
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)