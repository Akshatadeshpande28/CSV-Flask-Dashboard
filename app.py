from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

df_global = None

@app.route("/", methods=["GET", "POST"])
def index():
    global df_global

    summary = None
    chart_path = None
    heatmap_path = None
    columns = None

    if request.method == "POST":

        # Upload file
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df_global = pd.read_csv(filepath)

        if df_global is not None:
            df = df_global
            columns = df.columns.tolist()

            # Summary
            summary = df.describe().to_html(classes="table table-striped")

            # Heatmap
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                plt.figure(figsize=(8, 6))
                sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
                heatmap_path = os.path.join(STATIC_FOLDER, "heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()

        # Column-based chart
        x_col = request.form.get("x_col")
        y_col = request.form.get("y_col")

        if df_global is not None and x_col and y_col:
            plt.figure()
            df_global.plot(kind="scatter", x=x_col, y=y_col)
            chart_path = os.path.join(STATIC_FOLDER, "chart.png")
            plt.savefig(chart_path)
            plt.close()

    return render_template(
        "index.html",
        summary=summary,
        chart=chart_path,
        heatmap=heatmap_path,
        columns=columns
    )

if __name__ == "__main__":
    app.run(debug=True)