from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import os


# =========================================================
# FLASK APP CONFIGURATION
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Keeps the uploaded dataframe available while the app runs
df_global = None

# Keeps the uploaded filename available for chart requests
dataset_name_global = None


# =========================================================
# AI-STYLE AUTOMATIC INSIGHTS
# =========================================================

def generate_ai_insights(df):

    insights = []

    rows = len(df)
    cols = len(df.columns)

    # Dataset size
    insights.append(
        f"📊 The dataset contains {rows:,} rows and {cols} columns."
    )

    # Missing values
    missing = int(df.isnull().sum().sum())

    if missing > 0:
        missing_percentage = (
            missing / (rows * cols) * 100
            if rows > 0 and cols > 0
            else 0
        )

        insights.append(
            f"⚠️ There are {missing:,} missing values "
            f"({missing_percentage:.2f}% of all values)."
        )
    else:
        insights.append(
            "✅ No missing values were detected."
        )

    # Duplicate rows
    duplicates = int(df.duplicated().sum())

    if duplicates > 0:
        insights.append(
            f"🔁 The dataset contains {duplicates:,} duplicate rows."
        )
    else:
        insights.append(
            "✅ No duplicate rows were detected."
        )

    # Numeric columns
    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) > 0:

        insights.append(
            f"🔢 {len(numeric.columns)} numeric columns are available "
            f"for statistical analysis."
        )

    # Categorical columns
    categorical = df.select_dtypes(exclude="number")

    if len(categorical.columns) > 0:

        insights.append(
            f"🏷️ {len(categorical.columns)} categorical/text columns "
            f"were detected."
        )

    # Strongest correlation
    if len(numeric.columns) >= 2:

        corr = numeric.corr()

        correlation_pairs = corr.where(
            ~pd.DataFrame(
                False,
                index=corr.index,
                columns=corr.columns
            )
        )

        correlations = corr.unstack()

        # Remove same-column correlations
        correlations = correlations[
            correlations.index.get_level_values(0)
            != correlations.index.get_level_values(1)
        ]

        correlations = correlations.dropna()

        if not correlations.empty:

            # Find strongest relationship using absolute correlation
            strongest_index = correlations.abs().idxmax()

            strongest_value = correlations.loc[
                strongest_index
            ]

            col1 = strongest_index[0]
            col2 = strongest_index[1]

            insights.append(
                f"📈 The strongest numeric relationship is between "
                f"{col1} and {col2} "
                f"(correlation: {strongest_value:.2f})."
            )

    return " ".join(insights)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    global df_global
    global dataset_name_global

    # -----------------------------------------------------
    # Default values
    # -----------------------------------------------------

    summary = None
    preview = None

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

    missing_table = None
    duplicate_rows = None

    numeric_count = None
    categorical_count = None

    error = None

    selected_x = None
    selected_y = None
    selected_chart = None

    # -----------------------------------------------------
    # POST REQUEST
    # -----------------------------------------------------

    if request.method == "POST":

        try:

            # =================================================
            # CSV UPLOAD
            # =================================================

            file = request.files.get("file")

            if file and file.filename != "":

                # Validate file extension
                if not file.filename.lower().endswith(".csv"):

                    raise ValueError(
                        "Please upload a valid CSV file."
                    )

                dataset_name_global = file.filename

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    file.filename
                )

                file.save(filepath)

                # Read CSV
                df_global = pd.read_csv(filepath)

            # =================================================
            # DATASET ANALYSIS
            # =================================================

            if df_global is not None:

                df = df_global

                dataset_name = dataset_name_global

                # ---------------------------------------------
                # Basic dataset information
                # ---------------------------------------------

                columns = df.columns.tolist()

                rows = len(df)

                cols = len(df.columns)

                missing = int(
                    df.isnull().sum().sum()
                )

                duplicates = int(
                    df.duplicated().sum()
                )

                # Memory usage in KB
                memory = round(
                    df.memory_usage(
                        deep=True
                    ).sum() / 1024,
                    2
                )

                # ---------------------------------------------
                # Column types
                # ---------------------------------------------

                numeric_count = len(
                    df.select_dtypes(
                        include="number"
                    ).columns
                )

                categorical_count = len(
                    df.select_dtypes(
                        exclude="number"
                    ).columns
                )

                # =================================================
                # DATASET PREVIEW
                # =================================================

                preview = (
                    df.head(20)
                    .to_html(
                        classes=(
                            "table table-bordered "
                            "table-hover table-striped"
                        ),
                        index=False,
                        border=0
                    )
                )

                # =================================================
                # SUMMARY STATISTICS
                # =================================================

                try:

                    summary_df = (
                        df.describe(
                            include="all"
                        )
                        .fillna("")
                    )

                    summary = (
                        summary_df.to_html(
                            classes=(
                                "table table-striped "
                                "table-hover"
                            ),
                            border=0
                        )
                    )

                except Exception:

                    summary = None

                # =================================================
                # MISSING VALUE ANALYSIS
                # =================================================

                if rows > 0:

                    missing_df = pd.DataFrame({

                        "Column":
                            df.columns,

                        "Missing Values":
                            df.isnull().sum().values,

                        "Missing Percentage":
                            (
                                df.isnull().sum().values
                                / rows
                                * 100
                            ).round(2)

                    })

                else:

                    missing_df = pd.DataFrame({

                        "Column":
                            df.columns,

                        "Missing Values":
                            0,

                        "Missing Percentage":
                            0

                    })

                missing_df = missing_df.sort_values(
                    by="Missing Values",
                    ascending=False
                )

                missing_table = (
                    missing_df.to_html(
                        classes=(
                            "table table-striped "
                            "table-hover"
                        ),
                        index=False,
                        border=0
                    )
                )

                # =================================================
                # DUPLICATE ROW ANALYSIS
                # =================================================

                if duplicates > 0:

                    duplicate_df = (
                        df[df.duplicated()]
                        .head(20)
                    )

                    duplicate_rows = (
                        duplicate_df.to_html(
                            classes=(
                                "table table-striped "
                                "table-hover"
                            ),
                            index=False,
                            border=0
                        )
                    )

                # =================================================
                # AUTOMATIC INSIGHTS
                # =================================================

                ai_insights = generate_ai_insights(
                    df
                )

                # =================================================
                # CORRELATION HEATMAP
                # =================================================

                numeric_cols = (
                    df.select_dtypes(
                        include="number"
                    )
                )

                if len(numeric_cols.columns) >= 2:

                    correlation_matrix = (
                        numeric_cols.corr()
                    )

                    plt.figure(
                        figsize=(10, 7)
                    )

                    sns.heatmap(
                        correlation_matrix,
                        annot=True,
                        cmap="RdYlBu",
                        linewidths=0.5,
                        fmt=".2f"
                    )

                    plt.title(
                        "Correlation Heatmap",
                        fontsize=14
                    )

                    plt.tight_layout()

                    heatmap_filename = (
                        "heatmap.png"
                    )

                    heatmap_file = os.path.join(
                        STATIC_FOLDER,
                        heatmap_filename
                    )

                    plt.savefig(
                        heatmap_file,
                        dpi=120,
                        bbox_inches="tight"
                    )

                    plt.close()

                    heatmap = (
                        "/static/"
                        + heatmap_filename
                    )

                # =================================================
                # INTERACTIVE PLOTLY CHARTS
                # =================================================

                selected_x = request.form.get(
                    "x_col"
                )

                selected_y = request.form.get(
                    "y_col"
                )

                selected_chart = request.form.get(
                    "chart_type"
                )

                if selected_chart and selected_x:

                    # Validate X column
                    if selected_x not in df.columns:

                        raise ValueError(
                            "Selected X-axis column "
                            "does not exist."
                        )

                    # ---------------------------------------------
                    # Scatter Plot
                    # ---------------------------------------------

                    if selected_chart == "scatter":

                        if (
                            selected_y
                            and selected_y
                            in df.columns
                        ):

                            fig = px.scatter(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=(
                                    f"{selected_x} "
                                    f"vs {selected_y}"
                                )
                            )

                        else:

                            raise ValueError(
                                "Please select a valid "
                                "Y-axis column."
                            )

                    # ---------------------------------------------
                    # Bar Chart
                    # ---------------------------------------------

                    elif selected_chart == "bar":

                        if (
                            selected_y
                            and selected_y
                            in df.columns
                        ):

                            fig = px.bar(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=(
                                    f"{selected_y} "
                                    f"by {selected_x}"
                                )
                            )

                        else:

                            raise ValueError(
                                "Please select a valid "
                                "Y-axis column."
                            )

                    # ---------------------------------------------
                    # Line Chart
                    # ---------------------------------------------

                    elif selected_chart == "line":

                        if (
                            selected_y
                            and selected_y
                            in df.columns
                        ):

                            fig = px.line(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=(
                                    f"{selected_y} "
                                    f"Trend by "
                                    f"{selected_x}"
                                )
                            )

                        else:

                            raise ValueError(
                                "Please select a valid "
                                "Y-axis column."
                            )

                    # ---------------------------------------------
                    # Histogram
                    # ---------------------------------------------

                    elif selected_chart == "hist":

                        fig = px.histogram(
                            df,
                            x=selected_x,
                            title=(
                                f"Distribution of "
                                f"{selected_x}"
                            )
                        )

                    # ---------------------------------------------
                    # Pie Chart
                    # ---------------------------------------------

                    elif selected_chart == "pie":

                        value_counts = (
                            df[selected_x]
                            .value_counts()
                            .head(15)
                            .reset_index()
                        )

                        value_counts.columns = [
                            selected_x,
                            "Count"
                        ]

                        fig = px.pie(
                            value_counts,
                            names=selected_x,
                            values="Count",
                            title=(
                                f"Distribution of "
                                f"{selected_x}"
                            )
                        )

                    # ---------------------------------------------
                    # Box Plot
                    # ---------------------------------------------

                    elif selected_chart == "box":

                        if (
                            selected_y
                            and selected_y
                            in df.columns
                        ):

                            fig = px.box(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=(
                                    f"{selected_y} "
                                    f"by {selected_x}"
                                )
                            )

                        else:

                            fig = px.box(
                                df,
                                y=selected_x,
                                title=(
                                    f"Distribution of "
                                    f"{selected_x}"
                                )
                            )

                    else:

                        raise ValueError(
                            "Invalid chart type selected."
                        )

                    # ---------------------------------------------
                    # Chart Styling
                    # ---------------------------------------------

                    fig.update_layout(

                        template="plotly_white",

                        height=550,

                        margin=dict(
                            l=40,
                            r=40,
                            t=70,
                            b=40
                        ),

                        font=dict(
                            family="Arial"
                        )

                    )

                    plot_html = pio.to_html(
                        fig,
                        full_html=False,
                        config={
                            "responsive": True,
                            "displaylogo": False
                        }
                    )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except pd.errors.EmptyDataError:

            error = (
                "The uploaded CSV file is empty."
            )

        except pd.errors.ParserError:

            error = (
                "The CSV file could not be read. "
                "Please check its format."
            )

        except UnicodeDecodeError:

            error = (
                "The file encoding is not supported. "
                "Please save the CSV using UTF-8 encoding."
            )

        except Exception as e:

            error = str(e)

    # =========================================================
    # RENDER PAGE
    # =========================================================

    return render_template(

        "index.html",

        summary=summary,

        preview=preview,

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

        missing_table=missing_table,

        duplicate_rows=duplicate_rows,

        numeric_count=numeric_count,

        categorical_count=categorical_count,

        selected_x=selected_x,

        selected_y=selected_y,

        selected_chart=selected_chart,

        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )