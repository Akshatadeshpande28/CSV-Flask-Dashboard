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

ADMIN_EMAIL = os.environ.get(
    "ADMIN_EMAIL",
    "akshata.jd03@gmail.com"
).strip().lower()


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-change-this"
)

database_url = os.environ.get("DATABASE_URL")

# Allows local testing if DATABASE_URL is unavailable.
# Render will use your PostgreSQL DATABASE_URL.
if not database_url:
    database_url = "sqlite:///data_insight_pro.db"

# Some providers may return postgres:// instead of postgresql://
if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql://",
        1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Maximum CSV upload size: 25 MB
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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


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
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    last_login = db.Column(
        db.DateTime(timezone=True),
        nullable=True
    )

    def set_password(self, password):

        self.password_hash = generate_password_hash(
            password
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
        default=lambda: datetime.now(timezone.utc),
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
    def decorated_function(*args, **kwargs):

        if not current_user.is_authenticated:
            return redirect(
                url_for("login")
            )

        if not current_user.is_admin:
            abort(403)

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

    return pd.read_csv(path)


def load_filtered_dataframe():

    path = get_filtered_path()

    if os.path.exists(path):
        return pd.read_csv(path)

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
    cols = len(df.columns)

    insights.append(
        f"📊 The dataset contains {rows:,} rows "
        f"and {cols} columns."
    )

    missing = int(
        df.isnull().sum().sum()
    )

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

    duplicates = int(
        df.duplicated().sum()
    )

    if duplicates > 0:

        insights.append(
            f"🔁 {duplicates:,} duplicate rows were detected."
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
            f"🔢 {len(numeric.columns)} numeric columns "
            f"are available for statistical analysis."
        )

    if len(categorical.columns) > 0:

        insights.append(
            f"🏷️ {len(categorical.columns)} categorical/text "
            f"columns were detected."
        )

    if len(numeric.columns) >= 2:

        correlation_matrix = numeric.corr()

        correlations = correlation_matrix.unstack()

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
    # CORRELATION HEATMAP
    # =========================================================

    dashboard["heatmap"] = None

    try:
        numeric_df = df.select_dtypes(include="number")

        if len(numeric_df.columns) >= 2:

            correlation_matrix = numeric_df.corr()

            # Use Plotly instead of saving a Matplotlib image.
            heatmap_fig = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r"
            )

            heatmap_fig.update_layout(
                height=500,
                margin=dict(
                    l=40,
                    r=40,
                    t=70,
                    b=40
                )
            )

            dashboard["heatmap"] = pio.to_html(
                heatmap_fig,
                full_html=False,
                config={
                    "responsive": True,
                    "displaylogo": False
                }
            )

    except Exception as heatmap_error:
        print(
            "HEATMAP ERROR:",
            repr(heatmap_error)
        )
        dashboard["heatmap"] = None


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
            | column_text.str.contains(
                search_text,
                na=False,
                regex=False
            )
        )

    return (
        df[mask]
        .copy()
        .reset_index(drop=True)
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

    mean = series.mean()
    median = series.median()
    minimum = series.min()
    maximum = series.max()
    std_dev = series.std()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = series[
        (series < lower_bound)
        | (series > upper_bound)
    ]

    outlier_count = len(
        outliers
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

    skewness = series.skew()

    if pd.isna(skewness):

        distribution_text = (
            "Skewness could not be determined."
        )

    elif skewness > 1:

        distribution_text = (
            "The distribution is strongly right-skewed."
        )

    elif skewness > 0.5:

        distribution_text = (
            "The distribution is moderately right-skewed."
        )

    elif skewness < -1:

        distribution_text = (
            "The distribution is strongly left-skewed."
        )

    elif skewness < -0.5:

        distribution_text = (
            "The distribution is moderately left-skewed."
        )

    else:

        distribution_text = (
            "The distribution is approximately symmetric."
        )

    insight_parts = [
        f"{column} has an average value "
        f"of {mean:.2f} and a median of "
        f"{median:.2f}.",

        distribution_text
    ]

    if outlier_count > 0:

        outlier_percentage = (
            outlier_count
            / len(series)
            * 100
        )

        insight_parts.append(
            f"{outlier_count:,} potential outlier(s) "
            f"were identified using the IQR method "
            f"({outlier_percentage:.2f}% "
            f"of valid observations)."
        )

    else:

        insight_parts.append(
            "No potential outliers were identified "
            "using the IQR method."
        )

    if missing_count > 0:

        insight_parts.append(
            f"The column also contains "
            f"{missing_count:,} missing value(s)."
        )

    insight = " ".join(
        insight_parts
    )

    hist_fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="rug",
        title=f"Distribution of {column}"
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
        title=f"Outlier Analysis: {column}"
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
        "column": column,
        "type": "Numeric",
        "count": int(series.count()),
        "missing": missing_count,
        "unique": unique_count,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": round(minimum, 2),
        "max": round(maximum, 2),

        "std":
            round(std_dev, 2)
            if pd.notna(std_dev)
            else 0,

        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "outliers": outlier_count,

        "skewness":
            round(skewness, 2)
            if pd.notna(skewness)
            else 0,

        "insight": insight,
        "histogram": histogram_html,
        "boxplot": boxplot_html
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
        series.isnull().sum()
    )

    unique_count = int(
        non_null.nunique()
    )

    if non_null.empty:

        most_common = "N/A"
        most_common_count = 0

    else:

        mode = non_null.mode()

        most_common = (
            str(mode.iloc[0])
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
            title=f"Top Categories in {column}"
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
        f"{column} contains "
        f"{unique_count:,} unique "
        f"non-null value(s)."
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
            f"{missing_count:,} missing value(s)."
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
            category_counts
            .to_html(
                classes=(
                    "table table-striped "
                    "table-hover"
                ),
                index=False,
                border=0
            )
        )

    return {
        "column": column,
        "type": "Categorical",
        "count": int(non_null.count()),
        "missing": missing_count,
        "unique": unique_count,
        "most_common": most_common,
        "most_common_count": most_common_count,
        "insight": insight,
        "category_chart": category_html,
        "category_table": category_table
    }


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
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

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name or not email or not password:

            flash(
                "Please complete all required fields.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if "@" not in email:

            flash(
                "Please enter a valid email address.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
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
                "An account with this email already exists.",
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

        db.session.add(
            user
        )

        db.session.commit()

        flash(
            "Account created successfully. Please log in.",
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
    methods=["GET", "POST"]
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

            login_user(
                user
            )

            now = datetime.now(
                timezone.utc
            )

            user.last_login = now

            login_record = LoginHistory(
                user_id=user.id,
                login_time=now
            )

            db.session.add(
                login_record
            )

            db.session.commit()

            session[
                "login_history_id"
            ] = login_record.id

            flash(
                f"Welcome back, {user.name}!",
                "success"
            )

            next_page = request.args.get(
                "next"
            )

            # Avoid redirecting to external URLs.
            if (
                next_page
                and next_page.startswith("/")
                and not next_page.startswith("//")
            ):
                return redirect(next_page)

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
            and history.user_id
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
    methods=["GET", "POST"]
)
def index():

    # Public landing page for visitors; authenticated users see the dashboard.
    if not current_user.is_authenticated:
        return render_template("landing.html")

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

            file = request.files.get(
                "file"
            )

            action = request.form.get(
                "action"
            )

            # =================================================
            # FILE UPLOAD
            # =================================================

            if file and file.filename:

                filename = secure_filename(
                    file.filename
                )

                if not filename.lower().endswith(
                    ".csv"
                ):

                    raise ValueError(
                        "Please upload a valid CSV file."
                    )

                # Read directly from upload.
                df_global = pd.read_csv(
                    file
                )

                if df_global.empty:

                    raise ValueError(
                        "The uploaded CSV file contains no data."
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

                session["dataset_name"] = (
                    filename
                )

                flash(
                    "Dataset uploaded successfully.",
                    "success"
                )

            # =================================================
            # CHART
            # =================================================

            if action == "generate_chart":

                if df_filtered is None:

                    raise ValueError(
                        "Please upload a dataset first."
                    )

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
                    not selected_x
                    or selected_x
                    not in df_filtered.columns
                ):

                    raise ValueError(
                        "Please select a valid X-axis column."
                    )

                chart_df = df_filtered

                if selected_chart == "scatter":

                    if (
                        not selected_y
                        or selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid Y-axis column."
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

                elif selected_chart == "bar":

                    if (
                        not selected_y
                        or selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid Y-axis column."
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

                elif selected_chart == "line":

                    if (
                        not selected_y
                        or selected_y
                        not in chart_df.columns
                    ):

                        raise ValueError(
                            "Please select a valid Y-axis column."
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

                elif selected_chart == "hist":

                    fig = px.histogram(
                        chart_df,
                        x=selected_x,
                        title=(
                            f"Distribution of "
                            f"{selected_x}"
                        )
                    )

                elif selected_chart == "pie":

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

                elif selected_chart == "box":

                    if (
                        selected_y
                        and selected_y
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

            # =================================================
            # EDA
            # =================================================

            elif action == "generate_eda":

                if df_filtered is None:

                    raise ValueError(
                        "Please upload a dataset first."
                    )

                selected_eda_column = (
                    request.form.get(
                        "eda_column"
                    )
                )

                if (
                    not selected_eda_column
                    or selected_eda_column
                    not in df_filtered.columns
                ):

                    raise ValueError(
                        "Please select a valid column for EDA."
                    )

                if pd.api.types.is_numeric_dtype(
                    df_filtered[
                        selected_eda_column
                    ]
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
                "The uploaded CSV file is empty."
            )

        except pd.errors.ParserError:

            error = (
                "The CSV file could not be read. "
                "Please check its format."
            )

        except UnicodeDecodeError:

            error = (
                "Please save the CSV file using UTF-8 encoding."
            )

        except Exception as e:

            error = str(e)

    # Reload in case upload changed them.
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
        try:
            dashboard = prepare_dashboard_data(df_global)
        except Exception as dashboard_error:
            import traceback

            print("=" * 80)
            print("DASHBOARD PREPARATION ERROR")
            print(repr(dashboard_error))
            traceback.print_exc()
            print("=" * 80)

            error = (
                "The dataset was uploaded, but the dashboard "
                "could not be prepared. Please check the server logs."
            )
    original_rows = None
    filtered_rows = None
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
# TRY SAMPLE DATASET
# =========================================================

@app.route("/load-sample")
@login_required
def load_sample():

    sample_df = pd.DataFrame(
        {
            "Date": [
                "2026-01-05", "2026-01-08", "2026-01-12", "2026-01-18",
                "2026-01-22", "2026-02-03", "2026-02-10", "2026-02-15",
                "2026-02-21", "2026-03-02", "2026-03-11", "2026-03-18"
            ],
            "Product": [
                "Laptop", "Monitor", "Keyboard", "Mouse",
                "Laptop", "Monitor", "Keyboard", "Mouse",
                "Laptop", "Monitor", "Keyboard", "Mouse"
            ],
            "Region": [
                "North", "South", "East", "West",
                "North", "South", "East", "West",
                "North", "South", "East", "West"
            ],
            "Sales": [
                72000, 31000, 8500, 4200,
                68000, 35500, 9200, 5100,
                81000, 33000, 10400, 5600
            ],
            "Profit": [
                10800, 4650, 1700, 1050,
                10200, 5325, 1840, 1275,
                12150, 4950, 2080, 1400
            ],
            "Units Sold": [
                8, 12, 25, 42,
                7, 14, 28, 51,
                9, 13, 31, 55
            ],
            "Discount": [
                5, 10, 0, 5,
                8, 12, 0, 5,
                6, 10, 0, 7
            ]
        }
    )

    save_original_dataframe(sample_df)
    save_filtered_dataframe(sample_df)
    session["dataset_name"] = "sample_sales_dataset.csv"

    flash(
        "Sample sales dataset loaded successfully. Explore the dashboard features below.",
        "success"
    )

    return redirect(url_for("index"))


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
            "Please upload a dataset first.",
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
        and filter_column
        in filtered.columns
    ):

        if pd.api.types.is_numeric_dtype(
            filtered[
                filter_column
            ]
        ):

            if min_value:

                try:

                    minimum = float(
                        min_value
                    )

                    filtered = filtered[
                        filtered[
                            filter_column
                        ] >= minimum
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
                        ] <= maximum
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
            "Please upload a dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    removed = (
        before - len(df)
    )

    save_original_dataframe(df)
    save_filtered_dataframe(df)

    if removed > 0:

        flash(
            f"{removed:,} duplicate row(s) removed.",
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
            "Please upload a dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    before = len(df)

    df = (
        df
        .dropna()
        .reset_index(
            drop=True
        )
    )

    removed = (
        before - len(df)
    )

    save_original_dataframe(df)
    save_filtered_dataframe(df)

    flash(
        f"{removed:,} row(s) containing "
        f"missing values were removed.",
        "success"
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# SMART DATA CLEANING HELPERS
# =========================================================

def _find_column(df, name):
    """Find a column by normalized name without changing the user's schema."""
    target = name.strip().lower().replace(" ", "_")
    for col in df.columns:
        normalized = str(col).strip().lower().replace(" ", "_")
        if normalized == target:
            return col
    return None


def clean_dataset_safely(df):
    """Clean common data-quality issues without inventing identifiers/dates."""
    df = df.copy()
    report = {
        "duplicates_removed": 0,
        "missing_filled": 0,
        "missing_flagged": 0,
        "standardized": 0,
        "type_converted": 0,
        "changes": []
    }

    # ---------------------------------------------------------
    # 1. Remove exact duplicate rows
    # ---------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    # ---------------------------------------------------------
    # 2. Trim whitespace from text columns
    # ---------------------------------------------------------
    for col in df.select_dtypes(include=["object", "string"]).columns:
        original = df[col].copy()
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        changed = int((original.fillna("__NA__") != df[col].fillna("__NA__")).sum())
        if changed:
            report["standardized"] += changed
            report["changes"].append(f"Trimmed whitespace in {col}: {changed}")

    # ---------------------------------------------------------
    # 3. Standardize common categorical fields
    # ---------------------------------------------------------
    gender_col = _find_column(df, "Gender")
    if gender_col:
        mapping = {
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female"
        }
        old = df[gender_col].copy()
        df[gender_col] = df[gender_col].apply(
            lambda x: mapping.get(str(x).strip().lower(), x)
            if pd.notna(x) else x
        )
        changed = int((old.fillna("__NA__") != df[gender_col].fillna("__NA__")).sum())
        report["standardized"] += changed
        if changed:
            report["changes"].append(f"Standardized {changed} Gender value(s)")

    status_col = _find_column(df, "Order_Status")
    if status_col:
        allowed = {"completed": "Completed", "pending": "Pending",
                   "delivered": "Delivered", "cancelled": "Cancelled"}
        old = df[status_col].copy()
        df[status_col] = df[status_col].apply(
            lambda x: allowed.get(str(x).strip().lower(), x)
            if pd.notna(x) else x
        )
        changed = int((old.fillna("__NA__") != df[status_col].fillna("__NA__")).sum())
        report["standardized"] += changed
        if changed:
            report["changes"].append(f"Standardized {changed} Order Status value(s)")

    city_col = _find_column(df, "City")
    if city_col:
        city_map = {"hyderbad": "Hyderabad"}
        old = df[city_col].copy()
        df[city_col] = df[city_col].apply(
            lambda x: city_map.get(str(x).strip().lower(), str(x).strip())
            if pd.notna(x) else x
        )
        changed = int((old.fillna("__NA__") != df[city_col].fillna("__NA__")).sum())
        report["standardized"] += changed
        if changed:
            report["changes"].append(f"Corrected {changed} City value(s)")

    product_col = _find_column(df, "Product")
    if product_col:
        product_map = {"notebook": "Notebook"}
        old = df[product_col].copy()
        df[product_col] = df[product_col].apply(
            lambda x: product_map.get(str(x).strip().lower(), str(x).strip())
            if pd.notna(x) else x
        )
        changed = int((old.fillna("__NA__") != df[product_col].fillna("__NA__")).sum())
        report["standardized"] += changed
        if changed:
            report["changes"].append(f"Standardized {changed} Product value(s)")

    category_col = _find_column(df, "Category")
    if category_col:
        old = df[category_col].copy()
        df[category_col] = df[category_col].apply(
            lambda x: str(x).strip().title() if pd.notna(x) else x
        )
        changed = int((old.fillna("__NA__") != df[category_col].fillna("__NA__")).sum())
        report["standardized"] += changed

    # ---------------------------------------------------------
    # 4. Convert Amount / monetary fields to numeric safely
    # ---------------------------------------------------------
    amount_col = _find_column(df, "Amount")
    if amount_col:
        old = df[amount_col].copy()
        converted = pd.to_numeric(
            df[amount_col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )
        changed_type = int((old.astype(str) != converted.astype(str)).sum())
        df[amount_col] = converted
        if changed_type:
            report["type_converted"] += changed_type
            report["changes"].append(f"Converted {amount_col} to numeric")

    # ---------------------------------------------------------
    # 5. Parse dates into one consistent ISO format
    # ---------------------------------------------------------
    date_col = _find_column(df, "Order_Date")
    if date_col:
        old = df[date_col].copy()
        def parse_one_date(value):
            if pd.isna(value) or str(value).strip() == "":
                return pd.NaT
            text = str(value).strip()
            formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d/%m/%Y",
                "%d/%m/%y",
                "%d-%b-%Y",
                "%d-%B-%Y"
            ]
            for fmt in formats:
                try:
                    return pd.Timestamp(datetime.strptime(text, fmt))
                except (ValueError, TypeError):
                    pass
            return pd.NaT

        parsed = df[date_col].apply(parse_one_date)
        df[date_col] = parsed.apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else pd.NA
        )
        # Keep truly missing/unparseable dates blank rather than inventing them.
        df.loc[parsed.isna(), date_col] = pd.NA
        changed = int((old.fillna("__NA__").astype(str) != df[date_col].fillna("__NA__").astype(str)).sum())
        if changed:
            report["standardized"] += changed
            report["changes"].append(f"Standardized {changed} date value(s)")

    # ---------------------------------------------------------
    # 6. Numeric missing values: median only for measurable fields
    # ---------------------------------------------------------
    protected_numeric = {
        str(c).strip().lower() for c in [
            _find_column(df, "Customer_ID"),
            _find_column(df, "Order_ID")
        ] if c
    }
    for col in df.select_dtypes(include="number").columns:
        if str(col).strip().lower() in protected_numeric:
            continue
        missing = int(df[col].isna().sum())
        if missing:
            median = df[col].median()
            if pd.notna(median):
                df[col] = df[col].fillna(median)
                report["missing_filled"] += missing
                report["changes"].append(f"Filled {missing} missing value(s) in {col} with median")

    # ---------------------------------------------------------
    # 7. Never copy another person's Email / ID / Date.
    #    Keep missing values and flag them for review.
    # ---------------------------------------------------------
    protected_text = [
        _find_column(df, "Customer_ID"),
        _find_column(df, "Order_ID"),
        _find_column(df, "Email"),
        _find_column(df, "Order_Date")
    ]
    for col in protected_text:
        if col and df[col].isna().any():
            missing = int(df[col].isna().sum())
            report["missing_flagged"] += missing
            report["changes"].append(f"Flagged {missing} missing value(s) in {col} for review")

    # ---------------------------------------------------------
    # 8. Other categorical fields: use Unknown, not another person's value
    # ---------------------------------------------------------
    protected = {c for c in protected_text if c}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col in protected:
            continue
        missing = int(df[col].isna().sum())
        if missing:
            df[col] = df[col].fillna("Unknown")
            report["missing_filled"] += missing
            report["changes"].append(f"Filled {missing} missing value(s) in {col} with Unknown")

    return df, report


# =========================================================
# FILL MISSING / SMART CLEAN
# =========================================================

@app.route(
    "/fill-missing",
    methods=["POST"]
)
@login_required
def fill_missing():

    df = load_original_dataframe()

    if df is None:
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for("index"))

    before_missing = int(df.isnull().sum().sum())
    cleaned_df, report = clean_dataset_safely(df)
    after_missing = int(cleaned_df.isnull().sum().sum())

    save_original_dataframe(cleaned_df)
    save_filtered_dataframe(cleaned_df)

    summary = (
        f"Smart cleaning completed: "
        f"{report['duplicates_removed']} duplicate(s) removed, "
        f"{report['standardized']} value(s) standardized, "
        f"{report['type_converted']} type conversion(s), "
        f"{report['missing_filled']} missing value(s) safely filled."
    )

    if report["missing_flagged"]:
        summary += (
            f" {report['missing_flagged']} missing identifier/date value(s) "
            f"were preserved and flagged instead of being invented."
        )

    flash(summary, "success")
    return redirect(url_for("index"))


# =========================================================
# DOWNLOAD CLEANED CSV
# =========================================================


# =========================================================

@app.route("/download-csv")
@login_required
def download_csv():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    buffer = BytesIO()

    buffer.write(
        df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )
    )

    buffer.seek(0)

    name = os.path.splitext(
        current_dataset_name()
    )[0]

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

@app.route("/download-excel")
@login_required
def download_excel():

    df = (
        load_original_dataframe()
    )

    if df is None:

        flash(
            "Please upload a dataset first.",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    buffer = BytesIO()

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

    name = os.path.splitext(
        current_dataset_name()
    )[0]

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

@app.route("/download-filtered")
@login_required
def download_filtered():

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

    buffer = BytesIO()

    buffer.write(
        df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )
    )

    buffer.seek(0)

    name = os.path.splitext(
        current_dataset_name()
    )[0]

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=(
            f"{name}_filtered.csv"
        )
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
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
        login_history=login_history,
        total_users=total_users,
        total_logins=total_logins
    )


# =========================================================
# 403
# =========================================================

@app.errorhandler(403)
def forbidden(error):

    flash(
        "You do not have permission to access that page.",
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

    admin_email = ADMIN_EMAIL

    print(
        f"ADMIN_EMAIL configured as: {admin_email}"
    )

    admin_user = User.query.filter(
        db.func.lower(User.email) == admin_email
    ).first()

    if admin_user:

        if not admin_user.is_admin:
            admin_user.is_admin = True
            db.session.commit()
            print("ADMIN SUCCESS: User promoted to admin.")
        else:
            print("ADMIN SUCCESS: User is already an admin.")

    else:
        print("ADMIN INFO: No registered user matches ADMIN_EMAIL yet.")


# =========================================================
# RUN APP
# =========================================================

# =========================================================
# GLOBAL ERROR HANDLER - DEBUGGING
# =========================================================

@app.errorhandler(Exception)
def handle_unexpected_error(error):

    import traceback

    print("\n" + "=" * 80)
    print("DATA INSIGHT PRO - INTERNAL ERROR")
    print("=" * 80)
    print("ERROR:", repr(error))
    print("TRACEBACK:")
    traceback.print_exc()
    print("=" * 80 + "\n")

    return (
        "Data Insight Pro encountered an internal error. "
        "Please check the server logs.",
        500
    )
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