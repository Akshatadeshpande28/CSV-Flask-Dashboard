from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
    session,
    abort
)

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime, timezone
from functools import wraps

import os


# =========================================================
# ADMIN CONFIGURATION
# =========================================================

ADMIN_EMAIL = "akshata.jd03@gmail.com"


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-change-this"
)

database_url = os.environ.get("DATABASE_URL")

# Local fallback only.
# Render should use PostgreSQL through DATABASE_URL.
if not database_url:
    database_url = "sqlite:///data_insight_pro.db"

# Some PostgreSQL providers return postgres://
if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql://",
        1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum CSV size = 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# =========================================================
# DATABASE
# =========================================================

db = SQLAlchemy(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = (
    "Please log in to access Data Insight Pro."
)

login_manager.login_message_category = "warning"


# =========================================================
# FOLDERS
# =========================================================

UPLOAD_FOLDER = "uploads"

STATIC_FOLDER = "static"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    STATIC_FOLDER,
    exist_ok=True
)


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    is_admin = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(
            timezone.utc
        ),
        nullable=False
    )

    last_login = db.Column(
        db.DateTime(timezone=True),
        nullable=True
    )

    def set_password(self, password):

        self.password_hash = (
            generate_password_hash(
                password
            )
        )

    def check_password(self, password):

        return check_password_hash(
            self.password_hash,
            password
        )


# =========================================================
# LOGIN HISTORY MODEL
# =========================================================

class LoginHistory(db.Model):

    __tablename__ = "login_history"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    login_time = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(
            timezone.utc
        ),
        nullable=False
    )

    logout_time = db.Column(
        db.DateTime(timezone=True),
        nullable=True
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "login_records",
            lazy=True
        )
    )


# =========================================================
# LOAD LOGGED-IN USER
# =========================================================

@login_manager.user_loader
def load_user(user_id):

    try:

        return db.session.get(
            User,
            int(user_id)
        )

    except (ValueError, TypeError):

        return None


# =========================================================
# ADMIN PROTECTION
# =========================================================

def admin_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if not current_user.is_authenticated:

            return redirect(
                url_for("login")
            )

        # Only allow the configured admin email.
        if (
            current_user.email.strip().lower()
            != ADMIN_EMAIL.lower()
        ):

            abort(403)

        # Keep database role synchronized.
        if not current_user.is_admin:

            current_user.is_admin = True

            db.session.commit()

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# =========================================================
# USER DATASET HELPERS
# =========================================================

def get_user_folder():

    folder = os.path.join(
        UPLOAD_FOLDER,
        f"user_{current_user.id}"
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    return folder


def get_original_path():

    return os.path.join(
        get_user_folder(),
        "current_dataset.csv"
    )


def get_filtered_path():

    return os.path.join(
        get_user_folder(),
        "filtered_dataset.csv"
    )


def save_original_dataframe(df):

    df.to_csv(
        get_original_path(),
        index=False
    )


def save_filtered_dataframe(df):

    df.to_csv(
        get_filtered_path(),
        index=False
    )


def load_original_dataframe():

    path = get_original_path()

    if not os.path.exists(path):

        return None

    return pd.read_csv(
        path
    )


def load_filtered_dataframe():

    path = get_filtered_path()

    if os.path.exists(path):

        return pd.read_csv(
            path
        )

    return load_original_dataframe()


def current_dataset_name():

    return session.get(
        "dataset_name",
        "dataset.csv"
    )


# =========================================================
# PLOTLY HELPER
# =========================================================

def plot_to_html(fig):

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        )
    )

    return pio.to_html(
        fig,
        full_html=False,
        config={
            "responsive": True,
            "displaylogo": False
        }
    )


# =========================================================
# SMART DATA INSIGHTS
# =========================================================

def generate_ai_insights(df):

    insights = []

    rows = len(df)

    cols = len(
        df.columns
    )

    insights.append(
        f"📊 The dataset contains "
        f"{rows:,} rows and "
        f"{cols} columns."
    )

    missing = int(
        df.isnull()
        .sum()
        .sum()
    )

    if missing > 0:

        total_cells = (
            rows * cols
        )

        missing_percentage = (
            missing
            / total_cells
            * 100
            if total_cells > 0
            else 0
        )

        insights.append(
            f"⚠️ There are {missing:,} "
            f"missing values "
            f"({missing_percentage:.2f}% "
            f"of all values)."
        )

    else:

        insights.append(
            "✅ No missing values were detected."
        )

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates > 0:

        insights.append(
            f"🔁 {duplicates:,} duplicate "
            f"rows were detected."
        )

    else:

        insights.append(
            "✅ No duplicate rows were detected."
        )

    numeric = df.select_dtypes(
        include="number"
    )

    categorical = df.select_dtypes(
        exclude="number"
    )

    if len(numeric.columns) > 0:

        insights.append(
            f"🔢 {len(numeric.columns)} "
            f"numeric columns are available "
            f"for statistical analysis."
        )

    if len(categorical.columns) > 0:

        insights.append(
            f"🏷️ {len(categorical.columns)} "
            f"categorical/text columns "
            f"were detected."
        )

    if len(numeric.columns) >= 2:

        correlation_matrix = (
            numeric.corr()
        )

        correlations = (
            correlation_matrix.unstack()
        )

        correlations = correlations[
            correlations.index.get_level_values(0)
            !=
            correlations.index.get_level_values(1)
        ]

        correlations = (
            correlations.dropna()
        )

        if not correlations.empty:

            strongest_index = (
                correlations
                .abs()
                .idxmax()
            )

            strongest_value = (
                correlations.loc[
                    strongest_index
                ]
            )

            col1 = (
                strongest_index[0]
            )

            col2 = (
                strongest_index[1]
            )

            insights.append(
                f"📈 The strongest numeric "
                f"relationship is between "
                f"{col1} and {col2} "
                f"(correlation: "
                f"{strongest_value:.2f})."
            )

    return " ".join(
        insights
    )


# =========================================================
# PREPARE DASHBOARD
# =========================================================

def prepare_dashboard_data(df):

    dashboard = {}

    dashboard["columns"] = (
        df.columns.tolist()
    )

    dashboard["rows"] = (
        len(df)
    )

    dashboard["cols"] = (
        len(df.columns)
    )

    dashboard["missing"] = int(
        df.isnull()
        .sum()
        .sum()
    )

    dashboard["duplicates"] = int(
        df.duplicated()
        .sum()
    )

    dashboard["memory"] = round(
        df.memory_usage(
            deep=True
        ).sum() / 1024,
        2
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    dashboard["numeric_count"] = (
        len(numeric_columns)
    )

    dashboard["categorical_count"] = (
        len(categorical_columns)
    )

    dashboard["numeric_columns"] = (
        numeric_columns
    )

    dashboard["categorical_columns"] = (
        categorical_columns
    )

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

    missing_counts = (
        df.isnull().sum()
    )

    if len(df) > 0:

        missing_percentage = (
            missing_counts
            / len(df)
            * 100
        ).round(2)

    else:

        missing_percentage = (
            pd.Series(
                0,
                index=df.columns
            )
        )

    missing_df = pd.DataFrame(
        {
            "Column":
                df.columns,

            "Missing Values":
                missing_counts.values,

            "Missing Percentage (%)":
                missing_percentage.values
        }
    )

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

    dashboard[
        "duplicate_rows"
    ] = None

    if dashboard[
        "duplicates"
    ] > 0:

        duplicate_df = (
            df[
                df.duplicated()
            ]
            .head(20)
        )

        dashboard[
            "duplicate_rows"
        ] = (
            duplicate_df.to_html(
                classes=(
                    "table table-striped "
                    "table-hover"
                ),
                index=False,
                border=0
            )
        )

    dashboard[
        "ai_insights"
    ] = generate_ai_insights(
        df
    )

    dashboard[
        "heatmap"
    ] = None

    numeric_df = (
        df.select_dtypes(
            include="number"
        )
    )

    if len(
        numeric_df.columns
    ) >= 2:

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
            f"heatmap_user_"
            f"{current_user.id}.png"
        )

        heatmap_path = (
            os.path.join(
                STATIC_FOLDER,
                heatmap_filename
            )
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
# AUTOMATIC DASHBOARD GENERATOR
# =========================================================

def generate_automatic_dashboard(df):

    charts = []

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    # =====================================================
    # KPI VALUES
    # =====================================================

    total_rows = len(df)

    total_columns = len(
        df.columns
    )

    missing_values = int(
        df.isnull()
        .sum()
        .sum()
    )

    duplicate_rows = int(
        df.duplicated()
        .sum()
    )

    # =====================================================
    # CHART 1
    # Numeric Distribution
    # =====================================================

    if numeric_columns:

        column = numeric_columns[0]

        try:

            chart_df = df[
                [column]
            ].dropna()

            fig = px.histogram(
                chart_df,
                x=column,
                nbins=30,
                title=(
                    f"Distribution of {column}"
                )
            )

            charts.append({
                "title":
                    f"Distribution of {column}",

                "html":
                    plot_to_html(fig)
            })

        except Exception as e:

            print(
                "Automatic chart 1 error:",
                e
            )

    # =====================================================
    # CHART 2
    # Category vs Numeric
    # =====================================================

    if (
        categorical_columns
        and numeric_columns
    ):

        category = (
            categorical_columns[0]
        )

        value = (
            numeric_columns[0]
        )

        try:

            grouped = (
                df.groupby(
                    category,
                    dropna=False
                )[value]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .reset_index()
            )

            if not grouped.empty:

                grouped[
                    category
                ] = (
                    grouped[
                        category
                    ]
                    .astype(str)
                )

                fig = px.bar(
                    grouped,
                    x=category,
                    y=value,
                    title=(
                        f"{value} by {category}"
                    )
                )

                charts.append({
                    "title":
                        f"{value} by {category}",

                    "html":
                        plot_to_html(fig)
                })

        except Exception as e:

            print(
                "Automatic chart 2 error:",
                e
            )

    # =====================================================
    # CHART 3
    # Numeric Relationship
    # =====================================================

    if len(
        numeric_columns
    ) >= 2:

        x_column = (
            numeric_columns[0]
        )

        y_column = (
            numeric_columns[1]
        )

        try:

            chart_df = df[
                [
                    x_column,
                    y_column
                ]
            ].dropna()

            fig = px.scatter(
                chart_df,
                x=x_column,
                y=y_column,
                title=(
                    f"{x_column} vs {y_column}"
                )
            )

            charts.append({
                "title":
                    f"{x_column} vs {y_column}",

                "html":
                    plot_to_html(fig)
            })

        except Exception as e:

            print(
                "Automatic chart 3 error:",
                e
            )

    # =====================================================
    # CHART 4
    # Top Categories
    # =====================================================

    if categorical_columns:

        category = (
            categorical_columns[0]
        )

        try:

            counts = (
                df[category]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                category,
                "Count"
            ]

            if not counts.empty:

                fig = px.bar(
                    counts,
                    x=category,
                    y="Count",
                    title=(
                        f"Top {category} Categories"
                    )
                )

                charts.append({
                    "title":
                        f"Top {category} Categories",

                    "html":
                        plot_to_html(fig)
                })

        except Exception as e:

            print(
                "Automatic chart 4 error:",
                e
            )

    # =====================================================
    # RETURN DASHBOARD
    # =====================================================

    return {

        "total_rows":
            total_rows,

        "total_columns":
            total_columns,

        "missing_values":
            missing_values,

        "duplicate_rows":
            duplicate_rows,

        "charts":
            charts
    }

# =========================================================
# SEARCH DATAFRAME
# =========================================================

def search_dataframe(
    df,
    search_text
):

    if not search_text:

        return df.copy()

    search_text = (
        str(search_text)
        .lower()
        .strip()
    )

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

        mask = (
            mask
            |
            column_text.str.contains(
                search_text,
                na=False,
                regex=False
            )
        )

    return (
        df[mask]
        .copy()
        .reset_index(
            drop=True
        )
    )


# =========================================================
# NUMERIC EDA
# =========================================================

def generate_numeric_eda(
    df,
    column
):

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if series.empty:

        return None

    mean = (
        series.mean()
    )

    median = (
        series.median()
    )

    minimum = (
        series.min()
    )

    maximum = (
        series.max()
    )

    std_dev = (
        series.std()
    )

    q1 = (
        series.quantile(
            0.25
        )
    )

    q3 = (
        series.quantile(
            0.75
        )
    )

    iqr = (
        q3 - q1
    )

    lower_bound = (
        q1
        - 1.5 * iqr
    )

    upper_bound = (
        q3
        + 1.5 * iqr
    )

    outliers = series[
        (series < lower_bound)
        |
        (series > upper_bound)
    ]

    outlier_count = (
        len(outliers)
    )

    missing_count = int(
        df[column]
        .isnull()
        .sum()
    )

    unique_count = int(
        df[column]
        .nunique(
            dropna=True
        )
    )

    skewness = (
        series.skew()
    )

    if pd.isna(
        skewness
    ):

        distribution_text = (
            "Skewness could not "
            "be determined."
        )

    elif skewness > 1:

        distribution_text = (
            "The distribution is "
            "strongly right-skewed."
        )

    elif skewness > 0.5:

        distribution_text = (
            "The distribution is "
            "moderately right-skewed."
        )

    elif skewness < -1:

        distribution_text = (
            "The distribution is "
            "strongly left-skewed."
        )

    elif skewness < -0.5:

        distribution_text = (
            "The distribution is "
            "moderately left-skewed."
        )

    else:

        distribution_text = (
            "The distribution is "
            "approximately symmetric."
        )

    insight_parts = [

        (
            f"{column} has an average "
            f"value of {mean:.2f} and "
            f"a median of {median:.2f}."
        ),

        distribution_text
    ]

    if outlier_count > 0:

        outlier_percentage = (
            outlier_count
            / len(series)
            * 100
        )

        insight_parts.append(
            f"{outlier_count:,} potential "
            f"outlier(s) were identified "
            f"using the IQR method "
            f"({outlier_percentage:.2f}% "
            f"of valid observations)."
        )

    else:

        insight_parts.append(
            "No potential outliers were "
            "identified using the IQR method."
        )

    if missing_count > 0:

        insight_parts.append(
            f"The column also contains "
            f"{missing_count:,} "
            f"missing value(s)."
        )

    insight = " ".join(
        insight_parts
    )

    hist_fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="rug",
        title=(
            f"Distribution of {column}"
        )
    )

    hist_fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency"
    )

    histogram_html = (
        plot_to_html(
            hist_fig
        )
    )

    box_fig = px.box(
        df,
        y=column,
        points="outliers",
        title=(
            f"Outlier Analysis: {column}"
        )
    )

    box_fig.update_layout(
        yaxis_title=column
    )

    boxplot_html = (
        plot_to_html(
            box_fig
        )
    )

    return {

        "column":
            column,

        "type":
            "Numeric",

        "count":
            int(series.count()),

        "missing":
            missing_count,

        "unique":
            unique_count,

        "mean":
            round(mean, 2),

        "median":
            round(median, 2),

        "min":
            round(minimum, 2),

        "max":
            round(maximum, 2),

        "std":
            round(std_dev, 2)
            if pd.notna(std_dev)
            else 0,

        "q1":
            round(q1, 2),

        "q3":
            round(q3, 2),

        "outliers":
            outlier_count,

        "skewness":
            round(skewness, 2)
            if pd.notna(skewness)
            else 0,

        "insight":
            insight,

        "histogram":
            histogram_html,

        "boxplot":
            boxplot_html
    }


# =========================================================
# CATEGORICAL EDA
# =========================================================

def generate_categorical_eda(
    df,
    column
):

    series = df[column]

    non_null = (
        series.dropna()
    )

    missing_count = int(
        series.isnull()
        .sum()
    )

    unique_count = int(
        non_null.nunique()
    )

    if non_null.empty:

        most_common = "N/A"

        most_common_count = 0

    else:

        mode = (
            non_null.mode()
        )

        most_common = (
            str(
                mode.iloc[0]
            )
            if not mode.empty
            else "N/A"
        )

        most_common_count = int(
            (
                non_null.astype(str)
                == most_common
            ).sum()
        )

    category_counts = (
        non_null
        .astype(str)
        .value_counts()
        .head(15)
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Count"
    ]

    if not category_counts.empty:

        category_fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title=(
                f"Top Categories in {column}"
            )
        )

        category_fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency"
        )

        category_html = (
            plot_to_html(
                category_fig
            )
        )

    else:

        category_html = None

    insight_parts = [

        (
            f"{column} contains "
            f"{unique_count:,} unique "
            f"non-null value(s)."
        )
    ]

    if most_common != "N/A":

        percentage = (
            most_common_count
            / len(non_null)
            * 100
            if len(non_null) > 0
            else 0
        )

        insight_parts.append(
            f'The most common value is '
            f'"{most_common}", appearing '
            f'{most_common_count:,} times '
            f'({percentage:.2f}% of '
            f'non-null records).'
        )

    if missing_count > 0:

        insight_parts.append(
            f"The column contains "
            f"{missing_count:,} "
            f"missing value(s)."
        )

    else:

        insight_parts.append(
            "No missing values were detected "
            "in this column."
        )

    insight = " ".join(
        insight_parts
    )

    category_table = None

    if not category_counts.empty:

        category_table = (
            category_counts.to_html(
                classes=(
                    "table table-striped "
                    "table-hover"
                ),
                index=False,
                border=0
            )
        )

    return {

        "column":
            column,

        "type":
            "Categorical",

        "count":
            int(non_null.count()),

        "missing":
            missing_count,

        "unique":
            unique_count,

        "most_common":
            most_common,

        "most_common_count":
            most_common_count,

        "insight":
            insight,

        "category_chart":
            category_html,

        "category_table":
            category_table
    }


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=[
        "GET",
        "POST"
    ]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("index")
        )

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = (
            request.form.get(
                "confirm_password",
                ""
            )
        )

        if (
            not name
            or not email
            or not password
        ):

            flash(
                "Please complete all "
                "required fields.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if "@" not in email:

            flash(
                "Please enter a valid "
                "email address.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 8:

            flash(
                "Password must contain "
                "at least 8 characters.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        existing_user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )

        if existing_user:

            flash(
                "An account with this email "
                "already exists.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        user = User(
            name=name,
            email=email
        )

        user.set_password(
            password
        )

        # If this email is the configured admin,
        # create it as an admin immediately.
        if (
            email
            == ADMIN_EMAIL.lower()
        ):

            user.is_admin = True

        db.session.add(
            user
        )

        db.session.commit()

        flash(
            "Account created successfully. "
            "Please log in.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("index")
        )

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )

        if (
            user
            and user.check_password(
                password
            )
        ):

            # =================================================
            # ADMIN RECOGNITION
            # =================================================

            if (
                user.email
                .strip()
                .lower()
                ==
                ADMIN_EMAIL.lower()
            ):

                if not user.is_admin:

                    user.is_admin = True

                    db.session.commit()

                    print(
                        "ADMIN SUCCESS: "
                        "Admin account activated "
                        "during login."
                    )

            login_user(
                user
            )

            now = datetime.now(
                timezone.utc
            )

            user.last_login = (
                now
            )

            login_record = (
                LoginHistory(
                    user_id=user.id,
                    login_time=now
                )
            )

            db.session.add(
                login_record
            )

            db.session.commit()

            session[
                "login_history_id"
            ] = login_record.id

            flash(
                f"Welcome back, "
                f"{user.name}!",
                "success"
            )

            next_page = (
                request.args.get(
                    "next"
                )
            )

            if (
                next_page
                and
                next_page.startswith("/")
                and
                not next_page.startswith("//")
            ):

                return redirect(
                    next_page
                )

            return redirect(
                url_for("index")
            )

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    history_id = session.get(
        "login_history_id"
    )

    if history_id:

        history = db.session.get(
            LoginHistory,
            history_id
        )

        if (
            history
            and
            history.user_id
            == current_user.id
        ):

            history.logout_time = (
                datetime.now(
                    timezone.utc
                )
            )

            db.session.commit()

    logout_user()

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
@login_required
def index():

    error = None

    plot_html = None

    selected_x = None

    selected_y = None

    selected_chart = None

    eda_result = None

    selected_eda_column = None

    df_global = (
        load_original_dataframe()
    )

    df_filtered = (
        load_filtered_dataframe()
    )

    if request.method == "POST":

        try:

            file = (
                request.files.get(
                    "file"
                )
            )

            action = (
                request.form.get(
                    "action"
                )
            )

            # =============================================
            # FILE UPLOAD
            # =============================================

            if (
                file
                and file.filename
            ):

                filename = (
                    secure_filename(
                        file.filename
                    )
                )

                if not (
                    filename
                    .lower()
                    .endswith(".csv")
                ):

                    raise ValueError(
                        "Please upload a "
                        "valid CSV file."
                    )

                df_global = (
                    pd.read_csv(
                        file
                    )
                )

                if df_global.empty:

                    raise ValueError(
                        "The uploaded CSV file "
                        "contains no data."
                    )

                df_filtered = (
                    df_global.copy()
                )

                save_original_dataframe(
                    df_global
                )

                save_filtered_dataframe(
                    df_filtered
                )

                session[
                    "dataset_name"
                ] = filename

                flash(
                    "Dataset uploaded successfully.",
                    "success"
                )

            # =============================================
            # CHART
            # =============================================

            if action == "generate_chart":

                if df_filtered is None:

                    raise ValueError(
                        "Please upload a "
                        "dataset first."
                    )

                selected_x = (
                    request.form.get(
                        "x_col"
                    )
                )

                selected_y = (
                    request.form.get(
                        "y_col"
                    )
                )

                selected_chart = (
                    request.form.get(
                        "chart_type"
                    )
                )

                if (
                    not selected_x
                    or
                    selected_x
                    not in df_filtered.columns
                ):

                    raise ValueError(
                        "Please select a valid "
                        "X-axis column."
                    )

                chart_df = (
                    df_filtered
                )

                if (
                    selected_chart
                    == "scatter"
                ):

                    if (
                        not selected_y
                        or
                        selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid "
                            "Y-axis column."
                        )

                    fig = px.scatter(
                        chart_df,
                        x=selected_x,
                        y=selected_y,
                        title=(
                            f"{selected_x} vs "
                            f"{selected_y}"
                        )
                    )

                elif (
                    selected_chart
                    == "bar"
                ):

                    if (
                        not selected_y
                        or
                        selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid "
                            "Y-axis column."
                        )

                    fig = px.bar(
                        chart_df,
                        x=selected_x,
                        y=selected_y,
                        title=(
                            f"{selected_y} by "
                            f"{selected_x}"
                        )
                    )

                elif (
                    selected_chart
                    == "line"
                ):

                    if (
                        not selected_y
                        or
                        selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid "
                            "Y-axis column."
                        )

                    fig = px.line(
                        chart_df,
                        x=selected_x,
                        y=selected_y,
                        title=(
                            f"{selected_y} Trend "
                            f"by {selected_x}"
                        )
                    )

                elif (
                    selected_chart
                    == "hist"
                ):

                    fig = px.histogram(
                        chart_df,
                        x=selected_x,
                        title=(
                            f"Distribution of "
                            f"{selected_x}"
                        )
                    )

                elif (
                    selected_chart
                    == "pie"
                ):

                    value_counts = (
                        chart_df[
                            selected_x
                        ]
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

                elif (
                    selected_chart
                    == "box"
                ):

                    if (
                        selected_y
                        and
                        selected_y
                        in chart_df.columns
                    ):

                        fig = px.box(
                            chart_df,
                            x=selected_x,
                            y=selected_y,
                            title=(
                                f"{selected_y} by "
                                f"{selected_x}"
                            )
                        )

                    else:

                        fig = px.box(
                            chart_df,
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

                plot_html = (
                    plot_to_html(
                        fig
                    )
                )

            # =============================================
            # EDA
            # =============================================

            elif (
                action
                == "generate_eda"
            ):

                if df_filtered is None:

                    raise ValueError(
                        "Please upload a "
                        "dataset first."
                    )

                selected_eda_column = (
                    request.form.get(
                        "eda_column"
                    )
                )

                if (
                    not selected_eda_column
                    or
                    selected_eda_column
                    not in df_filtered.columns
                ):

                    raise ValueError(
                        "Please select a valid "
                        "column for EDA."
                    )

                if (
                    pd.api.types
                    .is_numeric_dtype(
                        df_filtered[
                            selected_eda_column
                        ]
                    )
                ):

                    eda_result = (
                        generate_numeric_eda(
                            df_filtered,
                            selected_eda_column
                        )
                    )

                else:

                    eda_result = (
                        generate_categorical_eda(
                            df_filtered,
                            selected_eda_column
                        )
                    )

        except pd.errors.EmptyDataError:

            error = (
                "The uploaded CSV "
                "file is empty."
            )

        except pd.errors.ParserError:

            error = (
                "The CSV file could not "
                "be read. Please check "
                "its format."
            )

        except UnicodeDecodeError:

            error = (
                "Please save the CSV file "
                "using UTF-8 encoding."
            )

        except Exception as e:

            error = str(e)

    df_global = (
        load_original_dataframe()
    )

    df_filtered = (
        load_filtered_dataframe()
    )

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

        dashboard = (
            prepare_dashboard_data(
                df_global
            )
        )

    original_rows = None

    filtered_rows = None

    filtered_preview = None

    if df_global is not None:

        original_rows = (
            len(df_global)
        )

    if df_filtered is not None:

        filtered_rows = (
            len(df_filtered)
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

        dataset_name=(
            session.get(
                "dataset_name"
            )
        ),

        plot_html=plot_html,

        selected_x=selected_x,

        selected_y=selected_y,

        selected_chart=selected_chart,

        original_rows=original_rows,

        filtered_rows=filtered_rows,

        filtered_preview=(
            filtered_preview
        ),

        eda_result=eda_result,

        selected_eda_column=(
            selected_eda_column
        ),

        error=error,

        status_message=None
    )


# =========================================================
# FILTER DATA
# =========================================================

@app.route(
    "/filter-data",
    methods=["POST"]
)
@login_required
def filter_data():

    df_global = (
        load_original_dataframe()
    )

    if df_global is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    filtered = (
        df_global.copy()
    )

    search_text = (
        request.form
        .get(
            "search_text",
            ""
        )
        .strip()
    )

    filter_column = (
        request.form
        .get(
            "filter_column",
            ""
        )
        .strip()
    )

    filter_value = (
        request.form
        .get(
            "filter_value",
            ""
        )
        .strip()
    )

    min_value = (
        request.form
        .get(
            "min_value",
            ""
        )
        .strip()
    )

    max_value = (
        request.form
        .get(
            "max_value",
            ""
        )
        .strip()
    )

    if search_text:

        filtered = (
            search_dataframe(
                filtered,
                search_text
            )
        )

    if (
        filter_column
        and
        filter_column
        in filtered.columns
    ):

        if (
            pd.api.types
            .is_numeric_dtype(
                filtered[
                    filter_column
                ]
            )
        ):

            if min_value:

                try:

                    minimum = float(
                        min_value
                    )

                    filtered = filtered[
                        filtered[
                            filter_column
                        ]
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
                        filtered[
                            filter_column
                        ]
                        <= maximum
                    ]

                except ValueError:

                    pass

        elif filter_value:

            filtered = filtered[
                filtered[
                    filter_column
                ]
                .astype(str)
                .str.contains(
                    filter_value,
                    case=False,
                    na=False,
                    regex=False
                )
            ]

    filtered = (
        filtered
        .copy()
        .reset_index(
            drop=True
        )
    )

    save_filtered_dataframe(
        filtered
    )

    flash(
        f"Filter applied. Showing "
        f"{len(filtered):,} of "
        f"{len(df_global):,} rows.",
        "info"
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
@login_required
def reset_filters():

    df_global = (
        load_original_dataframe()
    )

    if df_global is not None:

        save_filtered_dataframe(
            df_global
        )

        flash(
            "Filters reset successfully.",
            "success"
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
@login_required
def remove_duplicates():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    before = (
        len(df)
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    removed = (
        before
        - len(df)
    )

    save_original_dataframe(
        df
    )

    save_filtered_dataframe(
        df
    )

    if removed > 0:

        flash(
            f"{removed:,} duplicate "
            f"row(s) removed.",
            "success"
        )

    else:

        flash(
            "No duplicate rows were found.",
            "info"
        )

    return redirect(
        url_for("index")
    )


# =========================================================
# DROP MISSING
# =========================================================

@app.route(
    "/drop-missing",
    methods=["POST"]
)
@login_required
def drop_missing():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    before = (
        len(df)
    )

    df = (
        df
        .dropna()
        .reset_index(
            drop=True
        )
    )

    removed = (
        before
        - len(df)
    )

    save_original_dataframe(
        df
    )

    save_filtered_dataframe(
        df
    )

    flash(
        f"{removed:,} row(s) containing "
        f"missing values were removed.",
        "success"
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# FILL MISSING
# =========================================================

@app.route(
    "/fill-missing",
    methods=["POST"]
)
@login_required
def fill_missing():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    missing_before = int(
        df.isnull()
        .sum()
        .sum()
    )

    if missing_before == 0:

        flash(
            "No missing values were found.",
            "info"
        )

        return redirect(
            url_for("index")
        )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
    )

    for column in numeric_columns:

        if (
            df[column]
            .isnull()
            .any()
        ):

            median = (
                df[column]
                .median()
            )

            if pd.notna(
                median
            ):

                df[column] = (
                    df[column]
                    .fillna(
                        median
                    )
                )

    categorical_columns = (
        df.select_dtypes(
            exclude="number"
        )
        .columns
    )

    for column in categorical_columns:

        if (
            df[column]
            .isnull()
            .any()
        ):

            mode = (
                df[column]
                .mode()
            )

            if not mode.empty:

                df[column] = (
                    df[column]
                    .fillna(
                        mode.iloc[0]
                    )
                )

            else:

                df[column] = (
                    df[column]
                    .fillna(
                        "Unknown"
                    )
                )

    missing_after = int(
        df.isnull()
        .sum()
        .sum()
    )

    filled = (
        missing_before
        - missing_after
    )

    save_original_dataframe(
        df
    )

    save_filtered_dataframe(
        df
    )

    flash(
        f"{filled:,} missing value(s) "
        f"filled. Numeric columns used "
        f"median and categorical columns "
        f"used mode.",
        "success"
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
@login_required
def download_csv():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    buffer = (
        BytesIO()
    )

    buffer.write(
        df.to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    buffer.seek(0)

    name = (
        os.path.splitext(
            current_dataset_name()
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
# DOWNLOAD EXCEL
# =========================================================

@app.route(
    "/download-excel"
)
@login_required
def download_excel():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    buffer = (
        BytesIO()
    )

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    buffer.seek(0)

    name = (
        os.path.splitext(
            current_dataset_name()
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
# DOWNLOAD FILTERED
# =========================================================

@app.route(
    "/download-filtered"
)
@login_required
def download_filtered():

    df = (
        load_filtered_dataframe()
    )

    if df is None:

        flash(
            "Please upload a "
            "dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    buffer = (
        BytesIO()
    )

    buffer.write(
        df.to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    buffer.seek(0)

    name = (
        os.path.splitext(
            current_dataset_name()
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
# AUTOMATIC DASHBOARD
# =========================================================

@app.route(
    "/automatic-dashboard"
)
@login_required
def automatic_dashboard():

    df = (
        load_filtered_dataframe()
    )

    if df is None:

        flash(
            "Please upload a dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    dashboard = (
        generate_automatic_dashboard(
            df
        )
    )

    return render_template(
        "automatic_dashboard.html",

        dashboard=dashboard,

        dataset_name=(
            current_dataset_name()
        )
    )

# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route(
    "/admin"
)
@login_required
@admin_required
def admin():

    users = (
        User.query
        .order_by(
            User.created_at.desc()
        )
        .all()
    )

    login_history = (
        LoginHistory.query
        .order_by(
            LoginHistory.login_time.desc()
        )
        .limit(100)
        .all()
    )

    total_users = (
        User.query.count()
    )

    total_logins = (
        LoginHistory.query.count()
    )

    return render_template(
        "admin.html",

        users=users,

        login_history=(
            login_history
        ),

        total_users=(
            total_users
        ),

        total_logins=(
            total_logins
        )
    )


# =========================================================
# 403 ERROR
# =========================================================

@app.errorhandler(403)
def forbidden(error):

    flash(
        "You do not have permission "
        "to access that page.",
        "danger"
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# CREATE DATABASE TABLES + CONFIGURE ADMIN
# =========================================================

with app.app_context():

    db.create_all()

    admin_user = (
        User.query.filter(
            db.func.lower(
                User.email
            )
            ==
            ADMIN_EMAIL.lower()
        )
        .first()
    )

    if admin_user:

        if not admin_user.is_admin:

            admin_user.is_admin = True

            db.session.commit()

            print(
                "ADMIN SUCCESS: "
                "Admin account activated."
            )

        else:

            print(
                "ADMIN SUCCESS: "
                "Admin account already active."
            )

    else:

        print(
            "ADMIN INFO: "
            "Admin account has not "
            "been registered yet."
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
        debug=False
    )