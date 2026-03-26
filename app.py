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

# Store dataset globally (simple approach)
df_global = None


# 🤖 AI Insights Function
def generate_ai_insights(df):
    insights = []

    numeric_cols = df.select_dtypes(include="number")

    if not numeric_cols.empty:
        corr = numeric_cols.corr()

        # Strongest correlation
        max_corr = corr.unstack().sort_values(ascending=False)
        max_corr = max_corr[max_corr < 1]

        if not max_corr.empty:
            pair = max_corr.idxmax()
            insights.append(f"Strong relationship between {pair[0]} and {pair[1]}.")

    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"Dataset contains {missing} missing values.")

    # Dataset size
    insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    return " ".join(insights)


@app.route("/", methods=["GET", "POST"])
def index():
    global df_global

    summary = None
    chart_path = None
    heatmap_path = None
    columns = None
    ai_insights = None

    if request.method == "POST":

        # 📂 File Upload
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df_global = pd.read_csv(filepath)

        # If data exists
        if df_global is not None:
            df = df_global
            columns = df.columns.tolist()

            # 📊 Summary Stats
            summary = df.describe().to_html(classes="table table-striped")

            # 🔥 Heatmap
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                plt.figure(figsize=(8, 6))
                sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm")
                heatmap_path = os.path.join(STATIC_FOLDER, "heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()

            # 🤖 AI Insights
            ai_insights = generate_ai_insights(df)

        # 📈 Scatter Plot
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
        columns=columns,
        ai_insights=ai_insights
    )


if __name__ == "__main__":
    app.run(debug=True)