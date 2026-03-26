from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

df_global = None


# 🤖 AI Insights
def generate_ai_insights(df):
    insights = []

    numeric_cols = df.select_dtypes(include="number")

    if not numeric_cols.empty:
        corr = numeric_cols.corr()
        max_corr = corr.unstack().sort_values(ascending=False)
        max_corr = max_corr[max_corr < 1]

        if not max_corr.empty:
            pair = max_corr.idxmax()
            insights.append(f"Strong relationship between {pair[0]} and {pair[1]}.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"Dataset contains {missing} missing values.")

    insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    return " ".join(insights)


@app.route("/", methods=["GET", "POST"])
def index():
    global df_global

    summary = None
    columns = None
    heatmap_path = None
    ai_insights = None
    plot_html = None

    if request.method == "POST":

        # Upload
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

            # AI Insights
            ai_insights = generate_ai_insights(df)

            # Heatmap
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                plt.figure(figsize=(8, 6))
                sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
                heatmap_path = os.path.join(STATIC_FOLDER, "heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()

        # Chart Inputs
        x_col = request.form.get("x_col")
        y_col = request.form.get("y_col")
        chart_type = request.form.get("chart_type")

        if df_global is not None and chart_type:

            if chart_type == "scatter":
                fig = px.scatter(df_global, x=x_col, y=y_col)

            elif chart_type == "bar":
                fig = px.bar(df_global, x=x_col, y=y_col)

            elif chart_type == "line":
                fig = px.line(df_global, x=x_col, y=y_col)

            elif chart_type == "hist":
                fig = px.histogram(df_global, x=y_col)

            plot_html = pio.to_html(fig, full_html=False)

    return render_template(
        "index.html",
        summary=summary,
        columns=columns,
        heatmap=heatmap_path,
        ai_insights=ai_insights,
        plot_html=plot_html
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)