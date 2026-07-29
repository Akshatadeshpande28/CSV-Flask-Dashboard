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

# Master dataset after cleaning
df_global = None

# Temporary filtered dataset
df_filtered = None

dataset_name_global = None
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

    # Missing values
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
            f"({missing_percentage:.2f}% of all values)."
        )

    else:

        insights.append(
            "✅ No missing values were detected."
        )

    # Duplicates
    duplicates = int(df.duplicated().sum())

    if duplicates > 0:

        insights.append(
            f"🔁 {duplicates:,} duplicate rows were detected."
        )

    else:

        insights.append(
            "✅ No duplicate rows were detected."
        )

    # Numeric columns
    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) > 0:

        insights.append(
            f"🔢 {len(numeric.columns)} numeric columns "
            f"are available for analysis."
        )

    # Categorical columns
    categorical = df.select_dtypes(exclude="number")

    if len(categorical.columns) > 0:

        insights.append(
            f"🏷️ {len(categorical.columns)} categorical/text "
            f"columns were detected."
        )

    # Strongest correlation
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
                f"📈 The strongest numeric relationship is between "
                f"{col1} and {col2} "
                f"(correlation: {strongest_value:.2f})."
            )

    return " ".join(insights)


# =========================================================
# DASHBOARD DATA
# =========================================================

def prepare_dashboard_data(df):

    dashboard = {}

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
        df.memory_usage(deep=True).sum() / 1024,
        2
    )

    # Column types
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    dashboard["numeric_count"] = len(
        numeric_columns
    )

    dashboard["categorical_count"] = len(
        categorical_columns
    )

    dashboard["numeric_columns"] = numeric_columns

    dashboard["categorical_columns"] = categorical_columns

    # Dataset Preview
    dashboard["preview"] = (
        df.head(50)
        .to_html(
            classes=(
                "table table-bordered "
                "table-hover table-striped"
            ),
            index=False,
            border=0
        )
    )

    # Summary Statistics
    try:

        summary_df = (
            df.describe(include="all")
            .fillna("")
        )

        dashboard["summary"] = (
            summary_df.to_html(
                classes="table table-striped table-hover",
                border=0
            )
        )

    except Exception:

        dashboard["summary"] = None

    # Missing analysis
    rows = len(df)

    missing_counts = df.isnull().sum()

    if rows > 0:

        missing_percentage = (
            missing_counts / rows * 100
        ).round(2)

    else:

        missing_percentage = pd.Series(
            0,
            index=df.columns
        )

    missing_df = pd.DataFrame({

        "Column": df.columns,

        "Missing Values":
            missing_counts.values,

        "Missing Percentage (%)":
            missing_percentage.values

    })

    missing_df = missing_df.sort_values(
        "Missing Values",
        ascending=False
    )

    dashboard["missing_table"] = (
        missing_df.to_html(
            classes="table table-striped table-hover",
            index=False,
            border=0
        )
    )

    # Duplicate rows
    dashboard["duplicate_rows"] = None

    if dashboard["duplicates"] > 0:

        duplicate_df = (
            df[df.duplicated()]
            .head(20)
        )

        dashboard["duplicate_rows"] = (
            duplicate_df.to_html(
                classes="table table-striped table-hover",
                index=False,
                border=0
            )
        )

    # Insights
    dashboard["ai_insights"] = (
        generate_ai_insights(df)
    )

    # Heatmap
    dashboard["heatmap"] = None

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) >= 2:

        correlation_matrix = numeric_df.corr()

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

        heatmap_filename = "heatmap.png"

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
            "/static/" + heatmap_filename
        )

    return dashboard


# =========================================================
# SEARCH DATA
# =========================================================

def search_dataframe(df, search_text):

    if not search_text:
        return df.copy()

    search_text = str(
        search_text
    ).lower().strip()

    mask = pd.Series(
        False,
        index=df.index
    )

    for column in df.columns:

        column_text = (
            df[column]
            .astype(str)
            .str.lower()
        )

        mask = mask | column_text.str.contains(
            search_text,
            na=False,
            regex=False
        )

    return (
        df[mask]
        .copy()
        .reset_index(drop=True)
    )


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    global df_global
    global df_filtered
    global dataset_name_global
    global status_message

    error = None
    plot_html = None

    selected_x = None
    selected_y = None
    selected_chart = None

    if request.method == "POST":

        try:

            file = request.files.get("file")

            # -------------------------------------------------
            # FILE UPLOAD
            # -------------------------------------------------

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

                file.save(filepath)

                df_global = pd.read_csv(
                    filepath
                )

                df_filtered = df_global.copy()

                status_message = (
                    "Dataset uploaded successfully."
                )

            # -------------------------------------------------
            # CHART
            # -------------------------------------------------

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
                df_filtered is not None
                and selected_chart
                and selected_x
            ):

                chart_df = df_filtered

                if selected_x not in chart_df.columns:

                    raise ValueError(
                        "Selected X-axis column does not exist."
                    )

                if selected_chart == "scatter":

                    if (
                        selected_y
                        and selected_y in chart_df.columns
                    ):

                        fig = px.scatter(
                            chart_df,
                            x=selected_x,
                            y=selected_y,
                            title=f"{selected_x} vs {selected_y}"
                        )

                    else:

                        raise ValueError(
                            "Please select a valid Y-axis column."
                        )

                elif selected_chart == "bar":

                    if (
                        selected_y
                        and selected_y in chart_df.columns
                    ):

                        fig = px.bar(
                            chart_df,
                            x=selected_x,
                            y=selected_y,
                            title=f"{selected_y} by {selected_x}"
                        )

                    else:

                        raise ValueError(
                            "Please select a valid Y-axis column."
                        )

                elif selected_chart == "line":

                    if (
                        selected_y
                        and selected_y in chart_df.columns
                    ):

                        fig = px.line(
                            chart_df,
                            x=selected_x,
                            y=selected_y,
                            title=f"{selected_y} Trend by {selected_x}"
                        )

                    else:

                        raise ValueError(
                            "Please select a valid Y-axis column."
                        )

                elif selected_chart == "hist":

                    fig = px.histogram(
                        chart_df,
                        x=selected_x,
                        title=f"Distribution of {selected_x}"
                    )

                elif selected_chart == "pie":

                    value_counts = (
                        chart_df[selected_x]
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
                        title=f"Distribution of {selected_x}"
                    )

                elif selected_chart == "box":

                    if (
                        selected_y
                        and selected_y in chart_df.columns
                    ):

                        fig = px.box(
                            chart_df,
                            x=selected_x,
                            y=selected_y,
                            title=f"{selected_y} by {selected_x}"
                        )

                    else:

                        fig = px.box(
                            chart_df,
                            y=selected_x,
                            title=f"Distribution of {selected_x}"
                        )

                else:

                    raise ValueError(
                        "Invalid chart type selected."
                    )

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
                "Please save your CSV using UTF-8 encoding."
            )

        except Exception as e:

            error = str(e)

    # -----------------------------------------------------
    # Dashboard
    # -----------------------------------------------------

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
        "numeric_columns": [],
        "categorical_columns": [],
        "missing_table": None,
        "duplicate_rows": None

    }

    if df_global is not None:

        dashboard = prepare_dashboard_data(
            df_global
        )

    filtered_rows = None
    original_rows = None
    filtered_preview = None

    if df_global is not None:

        original_rows = len(
            df_global
        )

    if df_filtered is not None:

        filtered_rows = len(
            df_filtered
        )

        filtered_preview = (
            df_filtered
            .head(50)
            .to_html(
                classes=(
                    "table table-bordered "
                    "table-hover table-striped"
                ),
                index=False,
                border=0
            )
        )

    return render_template(

        "index.html",

        **dashboard,

        dataset_name=dataset_name_global,

        plot_html=plot_html,

        selected_x=selected_x,
        selected_y=selected_y,
        selected_chart=selected_chart,

        original_rows=original_rows,
        filtered_rows=filtered_rows,
        filtered_preview=filtered_preview,

        error=error,

        status_message=status_message
    )


# =========================================================
# FILTER DATA
# =========================================================

@app.route(
    "/filter-data",
    methods=["POST"]
)
def filter_data():

    global df_global
    global df_filtered
    global status_message

    if df_global is None:

        status_message = (
            "Please upload a dataset first."
        )

        return redirect(
            url_for("index")
        )

    filtered = df_global.copy()

    search_text = (
        request.form
        .get("search_text", "")
        .strip()
    )

    filter_column = (
        request.form
        .get("filter_column", "")
        .strip()
    )

    filter_value = (
        request.form
        .get("filter_value", "")
        .strip()
    )

    min_value = (
        request.form
        .get("min_value", "")
        .strip()
    )

    max_value = (
        request.form
        .get("max_value", "")
        .strip()
    )

    # -----------------------------------------------------
    # Global Search
    # -----------------------------------------------------

    if search_text:

        filtered = search_dataframe(
            filtered,
            search_text
        )

    # -----------------------------------------------------
    # Column filter
    # -----------------------------------------------------

    if (
        filter_column
        and filter_column in filtered.columns
    ):

        # Numeric
        if pd.api.types.is_numeric_dtype(
            filtered[filter_column]
        ):

            if min_value:

                try:

                    minimum = float(
                        min_value
                    )

                    filtered = filtered[
                        filtered[filter_column]
                        >= minimum
                    ]

                except ValueError:
                    pass

            if max_value:

                try:

                    maximum = float(
                        max_value
                    )

                    filtered = filtered[
                        filtered[filter_column]
                        <= maximum
                    ]

                except ValueError:
                    pass

        # Categorical
        elif filter_value:

            filtered = filtered[
                filtered[filter_column]
                .astype(str)
                .str.contains(
                    filter_value,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

    df_filtered = (
        filtered
        .copy()
        .reset_index(drop=True)
    )

    status_message = (
        f"Filter applied. "
        f"Showing {len(df_filtered):,} "
        f"of {len(df_global):,} rows."
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# RESET FILTERS
# =========================================================

@app.route(
    "/reset-filters",
    methods=["POST"]
)
def reset_filters():

    global df_global
    global df_filtered
    global status_message

    if df_global is not None:

        df_filtered = (
            df_global.copy()
        )

        status_message = (
            "Filters reset successfully."
        )

    return redirect(
        url_for("index")
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
    global df_filtered
    global status_message

    if df_global is None:

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

    removed = (
        before - len(df_global)
    )

    df_filtered = (
        df_global.copy()
    )

    status_message = (
        f"{removed} duplicate row(s) removed."
        if removed > 0
        else "No duplicate rows were found."
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# DROP MISSING ROWS
# =========================================================

@app.route(
    "/drop-missing",
    methods=["POST"]
)
def drop_missing():

    global df_global
    global df_filtered
    global status_message

    if df_global is None:

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

    removed = (
        before - len(df_global)
    )

    df_filtered = (
        df_global.copy()
    )

    status_message = (
        f"{removed} row(s) containing "
        f"missing values were removed."
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
    global df_filtered
    global status_message

    if df_global is None:

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

    # Numeric → Median
    numeric_columns = (
        df_global
        .select_dtypes(
            include="number"
        )
        .columns
    )

    for column in numeric_columns:

        if df_global[column].isnull().any():

            median = (
                df_global[column]
                .median()
            )

            if pd.notna(median):

                df_global[column] = (
                    df_global[column]
                    .fillna(median)
                )

    # Categorical → Mode
    categorical_columns = (
        df_global
        .select_dtypes(
            exclude="number"
        )
        .columns
    )

    for column in categorical_columns:

        if df_global[column].isnull().any():

            mode = (
                df_global[column]
                .mode()
            )

            if not mode.empty:

                df_global[column] = (
                    df_global[column]
                    .fillna(
                        mode.iloc[0]
                    )
                )

            else:

                df_global[column] = (
                    df_global[column]
                    .fillna("Unknown")
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

    df_filtered = (
        df_global.copy()
    )

    status_message = (
        f"{filled} missing value(s) filled. "
        f"Numeric values used median and "
        f"categorical values used mode."
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

    buffer = BytesIO()

    buffer.write(
        df_global
        .to_csv(index=False)
        .encode("utf-8")
    )

    buffer.seek(0)

    name = (
        os.path.splitext(
            dataset_name_global
            or "dataset"
        )[0]
    )

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=(
            f"{name}_cleaned.csv"
        )
    )


# =========================================================
# DOWNLOAD CLEANED EXCEL
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

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        df_global.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    buffer.seek(0)

    name = (
        os.path.splitext(
            dataset_name_global
            or "dataset"
        )[0]
    )

    return send_file(
        buffer,
        mimetype=(
            "application/"
            "vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        as_attachment=True,
        download_name=(
            f"{name}_cleaned.xlsx"
        )
    )


# =========================================================
# DOWNLOAD FILTERED CSV
# =========================================================

@app.route(
    "/download-filtered"
)
def download_filtered():

    global df_filtered
    global dataset_name_global

    if df_filtered is None:

        return redirect(
            url_for("index")
        )

    buffer = BytesIO()

    buffer.write(
        df_filtered
        .to_csv(index=False)
        .encode("utf-8")
    )

    buffer.seek(0)

    name = (
        os.path.splitext(
            dataset_name_global
            or "dataset"
        )[0]
    )

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=(
            f"{name}_filtered.csv"
        )
    )


# =========================================================
# RUN APP
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