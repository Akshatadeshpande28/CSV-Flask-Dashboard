from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file
)

import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO
import os


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# =========================================================
# GLOBAL DATA
# =========================================================

df_global = None
dataset_name_global = None

# Stores messages after cleaning operations
status_message = None


# =========================================================
# AUTOMATIC DATA INSIGHTS
# =========================================================

def generate_ai_insights(df):

    insights = []

    rows = len(df)
    cols = len(df.columns)

    insights.append(
        f"📊 The dataset contains {rows:,} rows and {cols} columns."
    )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    missing = int(df.isnull().sum().sum())

    if missing > 0:

        total_cells = rows * cols

        missing_percentage = (
            missing / total_cells * 100
            if total_cells > 0
            else 0
        )

        insights.append(
            f"⚠️ There are {missing:,} missing values "
            f"({missing_percentage:.2f}% of the dataset)."
        )

    else:

        insights.append(
            "✅ No missing values were detected."
        )

    # -----------------------------------------------------
    # Duplicates
    # -----------------------------------------------------

    duplicates = int(df.duplicated().sum())

    if duplicates > 0:

        insights.append(
            f"🔁 The dataset contains "
            f"{duplicates:,} duplicate rows."
        )

    else:

        insights.append(
            "✅ No duplicate rows were detected."
        )

    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    numeric = df.select_dtypes(
        include="number"
    )

    if len(numeric.columns) > 0:

        insights.append(
            f"🔢 {len(numeric.columns)} numeric columns "
            f"are available for statistical analysis."
        )

    # -----------------------------------------------------
    # Categorical columns
    # -----------------------------------------------------

    categorical = df.select_dtypes(
        exclude="number"
    )

    if len(categorical.columns) > 0:

        insights.append(
            f"🏷️ {len(categorical.columns)} categorical/text "
            f"columns were detected."
        )

    # -----------------------------------------------------
    # Strongest correlation
    # -----------------------------------------------------

    if len(numeric.columns) >= 2:

        corr = numeric.corr()

        correlations = corr.unstack()

        correlations = correlations[
            correlations.index.get_level_values(0)
            != correlations.index.get_level_values(1)
        ]

        correlations = correlations.dropna()

        if not correlations.empty:

            strongest_index = (
                correlations
                .abs()
                .idxmax()
            )

            strongest_value = correlations.loc[
                strongest_index
            ]

            col1 = strongest_index[0]
            col2 = strongest_index[1]

            insights.append(
                f"📈 The strongest numeric relationship "
                f"is between {col1} and {col2} "
                f"(correlation: {strongest_value:.2f})."
            )

    return " ".join(insights)


# =========================================================
# PREPARE DASHBOARD DATA
# =========================================================

def prepare_dashboard_data(df):

    dashboard = {}

    # -----------------------------------------------------
    # Basic information
    # -----------------------------------------------------

    dashboard["columns"] = df.columns.tolist()

    dashboard["rows"] = len(df)

    dashboard["cols"] = len(df.columns)

    dashboard["missing"] = int(
        df.isnull().sum().sum()
    )

    dashboard["duplicates"] = int(
        df.duplicated().sum()
    )

    dashboard["memory"] = round(
        df.memory_usage(
            deep=True
        ).sum() / 1024,
        2
    )

    # -----------------------------------------------------
    # Column type information
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns

    dashboard["numeric_count"] = len(
        numeric_columns
    )

    dashboard["categorical_count"] = len(
        categorical_columns
    )

    # -----------------------------------------------------
    # Dataset Preview
    # -----------------------------------------------------

    dashboard["preview"] = (
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

    # -----------------------------------------------------
    # Summary Statistics
    # -----------------------------------------------------

    try:

        summary_df = (
            df.describe(
                include="all"
            )
            .fillna("")
        )

        dashboard["summary"] = (
            summary_df.to_html(
                classes=(
                    "table table-striped "
                    "table-hover"
                ),
                border=0
            )
        )

    except Exception:

        dashboard["summary"] = None

    # -----------------------------------------------------
    # Missing Value Analysis
    # -----------------------------------------------------

    rows = len(df)

    if rows > 0:

        missing_counts = (
            df.isnull()
            .sum()
        )

        missing_percentage = (
            missing_counts
            / rows
            * 100
        ).round(2)

    else:

        missing_counts = pd.Series(
            0,
            index=df.columns
        )

        missing_percentage = pd.Series(
            0,
            index=df.columns
        )

    missing_df = pd.DataFrame({

        "Column":
            df.columns,

        "Missing Values":
            missing_counts.values,

        "Missing Percentage (%)":
            missing_percentage.values

    })

    missing_df = (
        missing_df
        .sort_values(
            "Missing Values",
            ascending=False
        )
    )

    dashboard["missing_table"] = (
        missing_df.to_html(
            classes=(
                "table table-striped "
                "table-hover"
            ),
            index=False,
            border=0
        )
    )

    # -----------------------------------------------------
    # Duplicate Rows
    # -----------------------------------------------------

    duplicates = int(
        df.duplicated().sum()
    )

    dashboard["duplicate_rows"] = None

    if duplicates > 0:

        duplicate_df = (
            df[df.duplicated()]
            .head(20)
        )

        dashboard["duplicate_rows"] = (
            duplicate_df.to_html(
                classes=(
                    "table table-striped "
                    "table-hover"
                ),
                index=False,
                border=0
            )
        )

    # -----------------------------------------------------
    # Insights
    # -----------------------------------------------------

    dashboard["ai_insights"] = (
        generate_ai_insights(df)
    )

    # -----------------------------------------------------
    # Correlation Heatmap
    # -----------------------------------------------------

    dashboard["heatmap"] = None

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) >= 2:

        correlation_matrix = (
            numeric_df.corr()
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

        heatmap_path = os.path.join(
            STATIC_FOLDER,
            heatmap_filename
        )

        plt.savefig(
            heatmap_path,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close()

        dashboard["heatmap"] = (
            "/static/"
            + heatmap_filename
        )

    return dashboard


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    global df_global
    global dataset_name_global
    global status_message

    error = None
    plot_html = None

    selected_x = None
    selected_y = None
    selected_chart = None

    # -----------------------------------------------------
    # CSV UPLOAD / CHART REQUEST
    # -----------------------------------------------------

    if request.method == "POST":

        try:

            file = request.files.get(
                "file"
            )

            # =================================================
            # FILE UPLOAD
            # =================================================

            if file and file.filename:

                if not file.filename.lower().endswith(
                    ".csv"
                ):

                    raise ValueError(
                        "Please upload a valid CSV file."
                    )

                dataset_name_global = (
                    file.filename
                )

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    file.filename
                )

                file.save(
                    filepath
                )

                df_global = pd.read_csv(
                    filepath
                )

                status_message = (
                    "Dataset uploaded successfully."
                )

            # =================================================
            # CHART GENERATION
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

            if (
                df_global is not None
                and selected_chart
                and selected_x
            ):

                df = df_global

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
                                f"Trend by {selected_x}"
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
                        "Invalid chart type."
                    )

                # ---------------------------------------------
                # Plotly styling
                # ---------------------------------------------

                fig.update_layout(

                    template="plotly_white",

                    height=550,

                    margin=dict(
                        l=40,
                        r=40,
                        t=70,
                        b=40
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
                "The CSV encoding is not supported. "
                "Please save the file using UTF-8."
            )

        except Exception as e:

            error = str(e)

    # =====================================================
    # PREPARE DASHBOARD
    # =====================================================

    dashboard = {

        "summary": None,
        "preview": None,
        "columns": None,
        "heatmap": None,
        "ai_insights": None,
        "rows": None,
        "cols": None,
        "missing": None,
        "duplicates": None,
        "memory": None,
        "numeric_count": None,
        "categorical_count": None,
        "missing_table": None,
        "duplicate_rows": None

    }

    if df_global is not None:

        dashboard = (
            prepare_dashboard_data(
                df_global
            )
        )

    return render_template(

        "index.html",

        summary=dashboard["summary"],

        preview=dashboard["preview"],

        columns=dashboard["columns"],

        heatmap=dashboard["heatmap"],

        ai_insights=dashboard[
            "ai_insights"
        ],

        rows=dashboard["rows"],

        cols=dashboard["cols"],

        missing=dashboard["missing"],

        duplicates=dashboard[
            "duplicates"
        ],

        memory=dashboard["memory"],

        numeric_count=dashboard[
            "numeric_count"
        ],

        categorical_count=dashboard[
            "categorical_count"
        ],

        missing_table=dashboard[
            "missing_table"
        ],

        duplicate_rows=dashboard[
            "duplicate_rows"
        ],

        dataset_name=dataset_name_global,

        plot_html=plot_html,

        selected_x=selected_x,

        selected_y=selected_y,

        selected_chart=selected_chart,

        error=error,

        status_message=status_message
    )


# =========================================================
# REMOVE DUPLICATES
# =========================================================

@app.route(
    "/remove-duplicates",
    methods=["POST"]
)
def remove_duplicates():

    global df_global
    global status_message

    if df_global is None:

        status_message = (
            "Please upload a dataset first."
        )

        return redirect(
            url_for("index")
        )

    before = len(
        df_global
    )

    df_global = (
        df_global
        .drop_duplicates()
        .reset_index(drop=True)
    )

    after = len(
        df_global
    )

    removed = (
        before - after
    )

    if removed > 0:

        status_message = (
            f"{removed} duplicate row(s) "
            f"removed successfully."
        )

    else:

        status_message = (
            "No duplicate rows were found."
        )

    return redirect(
        url_for("index")
    )


# =========================================================
# DROP ROWS WITH MISSING VALUES
# =========================================================

@app.route(
    "/drop-missing",
    methods=["POST"]
)
def drop_missing():

    global df_global
    global status_message

    if df_global is None:

        status_message = (
            "Please upload a dataset first."
        )

        return redirect(
            url_for("index")
        )

    before = len(
        df_global
    )

    df_global = (
        df_global
        .dropna()
        .reset_index(drop=True)
    )

    after = len(
        df_global
    )

    removed = (
        before - after
    )

    if removed > 0:

        status_message = (
            f"{removed} row(s) containing "
            f"missing values were removed."
        )

    else:

        status_message = (
            "No rows with missing values "
            "were found."
        )

    return redirect(
        url_for("index")
    )


# =========================================================
# FILL MISSING VALUES
# =========================================================

@app.route(
    "/fill-missing",
    methods=["POST"]
)
def fill_missing():

    global df_global
    global status_message

    if df_global is None:

        status_message = (
            "Please upload a dataset first."
        )

        return redirect(
            url_for("index")
        )

    missing_before = int(
        df_global
        .isnull()
        .sum()
        .sum()
    )

    if missing_before == 0:

        status_message = (
            "No missing values were found."
        )

        return redirect(
            url_for("index")
        )

    # -----------------------------------------------------
    # Numeric columns → Median
    # -----------------------------------------------------

    numeric_columns = (
        df_global
        .select_dtypes(
            include="number"
        )
        .columns
    )

    for column in numeric_columns:

        if df_global[column].isnull().any():

            median_value = (
                df_global[column]
                .median()
            )

            if pd.notna(
                median_value
            ):

                df_global[column] = (
                    df_global[column]
                    .fillna(
                        median_value
                    )
                )

    # -----------------------------------------------------
    # Categorical columns → Mode
    # -----------------------------------------------------

    categorical_columns = (
        df_global
        .select_dtypes(
            exclude="number"
        )
        .columns
    )

    for column in categorical_columns:

        if df_global[column].isnull().any():

            mode_values = (
                df_global[column]
                .mode()
            )

            if not mode_values.empty:

                df_global[column] = (
                    df_global[column]
                    .fillna(
                        mode_values.iloc[0]
                    )
                )

            else:

                df_global[column] = (
                    df_global[column]
                    .fillna(
                        "Unknown"
                    )
                )

    missing_after = int(
        df_global
        .isnull()
        .sum()
        .sum()
    )

    filled = (
        missing_before
        - missing_after
    )

    status_message = (
        f"{filled} missing value(s) "
        f"filled successfully. "
        f"Numeric columns used the median "
        f"and categorical columns used the mode."
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# DOWNLOAD CLEANED CSV
# =========================================================

@app.route(
    "/download-csv"
)
def download_csv():

    global df_global
    global dataset_name_global

    if df_global is None:

        return redirect(
            url_for("index")
        )

    csv_buffer = BytesIO()

    csv_data = (
        df_global
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    csv_buffer.write(
        csv_data
    )

    csv_buffer.seek(0)

    filename = (
        "cleaned_dataset.csv"
    )

    if dataset_name_global:

        original_name = (
            os.path.splitext(
                dataset_name_global
            )[0]
        )

        filename = (
            f"{original_name}_cleaned.csv"
        )

    return send_file(

        csv_buffer,

        mimetype="text/csv",

        as_attachment=True,

        download_name=filename
    )


# =========================================================
# DOWNLOAD EXCEL
# =========================================================

@app.route(
    "/download-excel"
)
def download_excel():

    global df_global
    global dataset_name_global

    if df_global is None:

        return redirect(
            url_for("index")
        )

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        df_global.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    excel_buffer.seek(0)

    filename = (
        "cleaned_dataset.xlsx"
    )

    if dataset_name_global:

        original_name = (
            os.path.splitext(
                dataset_name_global
            )[0]
        )

        filename = (
            f"{original_name}_cleaned.xlsx"
        )

    return send_file(

        excel_buffer,

        mimetype=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        as_attachment=True,

        download_name=filename
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