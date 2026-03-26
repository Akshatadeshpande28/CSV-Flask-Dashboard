from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    chart_path = None
    columns = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            columns = df.columns.tolist()

            # Summary statistics
            summary = df.describe().to_html(classes="table table-striped")

            # Plot numeric columns
            numeric_cols = df.select_dtypes(include="number").columns

            if len(numeric_cols) > 0:
                df[numeric_cols].hist(figsize=(10, 6))
                chart_path = os.path.join(STATIC_FOLDER, "chart.png")
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()

    # 👇 THIS IS THE IMPORTANT LINE
    return render_template("index.html", summary=summary, chart=chart_path, columns=columns)


if __name__ == "__main__":
    app.run(debug=True)