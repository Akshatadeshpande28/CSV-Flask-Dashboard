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


# ===============================
# AI Insights
# ===============================
def generate_ai_insights(df):

    insights = []

    rows = len(df)
    cols = len(df.columns)

    insights.append(
        f"📊 Dataset contains {rows} rows and {cols} columns."
    )

    missing = int(df.isnull().sum().sum())

    if missing:
        insights.append(
            f"⚠️ Dataset contains {missing} missing values."
        )
    else:
        insights.append(
            "✅ No missing values found."
        )

    duplicates = int(df.duplicated().sum())

    if duplicates:
        insights.append(
            f"🔁 Dataset contains {duplicates} duplicate rows."
        )
    else:
        insights.append(
            "✅ No duplicate rows detected."
        )

    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) >= 2:

        corr = numeric.corr()

        strongest = (
            corr.unstack()
            .drop_duplicates()
            .sort_values(ascending=False)
        )

        strongest = strongest[strongest < 1]

        if not strongest.empty:

            pair = strongest.idxmax()

            insights.append(
                f"📈 Strong correlation found between "
                f"{pair[0]} and {pair[1]}."
            )

    return " ".join(insights)


# ===============================
# Home Page
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():

    global df_global

    summary = None
    columns = None
    heatmap = None
    ai_insights = None
    plot_html = None

    rows = None
    cols = None
    missing = None
    duplicates = None
    memory = None
    dataset_name = None

    error = None

    if request.method == "POST":

        try:

            # ===============================
            # Upload CSV
            # ===============================

            file = request.files.get("file")

            if file and file.filename != "":

                dataset_name = file.filename

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    file.filename
                )

                file.save(filepath)

                df_global = pd.read_csv(filepath)

            # ===============================
            # Dataset Analysis
            # ===============================

            if df_global is not None:

                df = df_global

                columns = df.columns.tolist()

                rows = len(df)

                cols = len(columns)

                missing = int(
                    df.isnull().sum().sum()
                )

                duplicates = int(
                    df.duplicated().sum()
                )

                memory = round(
                    df.memory_usage(deep=True).sum() / 1024,
                    2
                )

                summary = (
                    df.describe(include="all")
                    .fillna("")
                    .to_html(
                        classes="table table-striped table-hover",
                        border=0
                    )
                )

                ai_insights = generate_ai_insights(df)

                # ===============================
                # Heatmap
                # ===============================

                numeric = df.select_dtypes(include="number")

                if not numeric.empty:

                    plt.figure(figsize=(10, 7))

                    sns.heatmap(
                        numeric.corr(),
                        annot=True,
                        cmap="RdYlBu",
                        linewidths=.5,
                        square=True
                    )

                    heatmap_file = os.path.join(
                        STATIC_FOLDER,
                        "heatmap.png"
                    )

                    plt.tight_layout()

                    plt.savefig(heatmap_file)

                    plt.close()

                    heatmap = "/" + heatmap_file.replace("\\", "/")

            # ===============================
            # Interactive Plotly Charts
            # ===============================

            x_col = request.form.get("x_col")
            y_col = request.form.get("y_col")
            chart_type = request.form.get("chart_type")

            if (
                df_global is not None
                and chart_type
                and x_col
            ):

                if chart_type == "scatter":

                    fig = px.scatter(
                        df_global,
                        x=x_col,
                        y=y_col
                    )

                elif chart_type == "bar":

                    fig = px.bar(
                        df_global,
                        x=x_col,
                        y=y_col
                    )

                elif chart_type == "line":

                    fig = px.line(
                        df_global,
                        x=x_col,
                        y=y_col
                    )

                elif chart_type == "hist":

                    fig = px.histogram(
                        df_global,
                        x=x_col
                    )

                else:

                    fig = px.scatter(
                        df_global,
                        x=x_col,
                        y=y_col
                    )

                fig.update_layout(
                    template="plotly_white",
                    height=550,
                    title="Interactive Data Visualization"
                )

                plot_html = pio.to_html(
                    fig,
                    full_html=False
                )

        except Exception as e:

            error = str(e)

    return render_template(

        "index.html",

        summary=summary,

        columns=columns,

        heatmap=heatmap,

        ai_insights=ai_insights,

        plot_html=plot_html,

        rows=rows,

        cols=cols,

        missing=missing,

        duplicates=duplicates,

        memory=memory,

        dataset_name=dataset_name,

        error=error

    )


# ===============================
# Run Flask
# ===============================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,