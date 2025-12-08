# -*- coding: utf-8 -*-
# Copyright 2024-2025 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import altair as alt
from altair.datasets import data
import polars as pl
import numpy as np
from streamlit.column_config import NumberColumn
import statsmodels.api as sm


st.set_page_config(
    page_title="Movies, movies, movies!", page_icon=":clapper:", layout="wide"
)

df = data.movies(engine="polars")
TITLE_COL = "Title"
IMDB_COL = "IMDB Rating"
RT_COL = "Rotten Tomatoes Rating"
DIRECTOR_COL = "Director"
HAS_RATINGS = pl.col(IMDB_COL).is_not_null() | pl.col(RT_COL).is_not_null()
GRID_WIDTH = 600
GRID_HEIGHT = 500
MARK_SIZE = 70
DIRECTOR_MARK_SIZE = 150

COLUMN_CONFIG = {
    TITLE_COL: st.column_config.TextColumn(pinned=True),
    IMDB_COL: st.column_config.ProgressColumn(
        min_value=0,
        max_value=10,
        color="auto",
        format="compact",
        width=100,
    ),
    RT_COL: st.column_config.ProgressColumn(
        min_value=0,
        max_value=100,
        color="auto",
        format="compact",
        width=100,
    ),
    "US Gross": st.column_config.NumberColumn(format="dollar"),
    "Worldwide Gross": st.column_config.NumberColumn(format="dollar"),
    "US DVD Sales": st.column_config.NumberColumn(format="dollar"),
    "Production Budget": st.column_config.NumberColumn(format="dollar"),
    # TODO: Fix Streamlit bug where options are required in order for color to work:
    "Distributor": st.column_config.MultiselectColumn(
        options=set(df.get_column("Distributor").to_list()), color="auto"
    ),
    "Director": st.column_config.MultiselectColumn(
        options=set(df.get_column("Director").to_list()), color="auto"
    ),
    "Major Genre": st.column_config.MultiselectColumn(
        options=set(df.get_column("Major Genre").to_list()), color="auto"
    ),
    "Creative Type": st.column_config.MultiselectColumn(
        options=set(df.get_column("Creative Type").to_list()), color="auto"
    ),
    "Source": st.column_config.MultiselectColumn(
        options=set(df.get_column("Source").to_list()), color="auto"
    ),
    "MPAA Rating": st.column_config.MultiselectColumn(
        options=set(df.get_column("MPAA Rating").to_list()), color="auto"
    ),
}

# -----------------------------------------------------------------------------
# Helpful functions


def draw_histogram(df, metric_name):
    # Ensure we don't have nulls for the main metric
    clean_df = df.drop_nulls(subset=[metric_name])

    st.altair_chart(
        alt.Chart(clean_df, height=200, width=200)
        .mark_bar(binSpacing=0)
        .encode(
            alt.X(
                metric_name,
                type="quantitative",
            ).bin(maxbins=20),
            alt.Y("count()").axis(None),
        )
    )


def draw_director_median_chart(title, data, x_col, y_col, x_domain, color_domain):
    data = data.drop_nulls(subset=[y_col])

    medians = (
        data.group_by(y_col)
        .agg(pl.col(x_col).median().alias("median_val"))
        .sort("median_val", descending=True)
    )

    sort_order = medians.get_column(y_col).to_list()

    base = alt.Chart(data).encode(
        alt.Y(f"{y_col}:N", sort=sort_order, title=None),
    )

    points = base.mark_point(filled=True, size=DIRECTOR_MARK_SIZE).encode(
        alt.X(f"{x_col}:Q", title=f"{x_col}").scale(zero=True, domain=x_domain),
        alt.Color(DIRECTOR_COL, type="nominal").scale(domain=color_domain).legend(None),
        alt.Shape(DIRECTOR_COL, type="nominal").scale(domain=color_domain).legend(None),
        tooltip=[y_col, x_col, TITLE_COL],
    )

    ticks = (
        alt.Chart(medians.to_pandas())
        .mark_tick(
            color="red",
            thickness=2,
        )
        .encode(
            alt.Y(f"{y_col}:N", sort=sort_order),
            alt.X("median_val:Q"),
            tooltip=[alt.Tooltip("median_val:Q", title="Median Rating")],
        )
    )

    st.subheader(title)
    st.altair_chart(points + ticks, width="stretch")


def perform_linear_regression(df, x_col, y_col, sigma_threshold):
    clean_df = df.drop_nulls([x_col, y_col])

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Degree 1 = Linear
    slope, intercept = np.polyfit(x, y, 1)

    predictions = (slope * x) + intercept
    residuals = y - predictions
    std_dev = np.std(residuals)

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    result_df = clean_df.with_columns(
        [
            pl.Series("Predicted", predictions),
            pl.Series("Upper Bound", upper_bound),
            pl.Series("Lower Bound", lower_bound),
            # Determine Status: Outlier if outside the bounds
            pl.when(
                (pl.col(y_col) > pl.Series(upper_bound))
                | (pl.col(y_col) < pl.Series(lower_bound))
            )
            .then(pl.lit("Outlier"))
            .otherwise(pl.lit("In Range"))
            .alias("Status"),
        ]
    )

    return result_df


def perform_loess_regression(df, x_col, y_col, sigma_threshold, frac=0.66):
    """
    Calculates LOESS regression, residuals, and outlier status using Polars and Statsmodels.

    Args:
        frac (float): The fraction of the data used when estimating each y-value.
                      Between 0 and 1. Defaults to 0.66 (standard).
    """
    # Sorting by x_col is mandatory for LOESS to align predictions correctly for plotting
    clean_df = df.drop_nulls([x_col, y_col]).sort(x_col)

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Returns an (n, 2) array: [sorted_x, fitted_y]
    lowess_result = sm.nonparametric.lowess(y, x, frac=frac)

    predictions = lowess_result[:, 1]

    residuals = y - predictions
    std_dev = np.std(residuals)

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    result_df = clean_df.with_columns(
        [
            pl.Series("Predicted", predictions),
            pl.Series("Upper Bound", upper_bound),
            pl.Series("Lower Bound", lower_bound),
            pl.when(
                (pl.col(y_col) > pl.Series(upper_bound))
                | (pl.col(y_col) < pl.Series(lower_bound))
            )
            .then(pl.lit("Outlier"))
            .otherwise(pl.lit("In Range"))
            .alias("Status"),
        ]
    )

    return result_df


# -----------------------------------------------------------------------------
# Draw app


st.title("Movies, movies, _movies!_")

st.space()

with st.container(width=GRID_WIDTH):
    """
    This is an analysis of the "Movies" dataset, provided by the University
    of Washington's Interactive Data Lab (IDL) and used in several of their
    [**Vega-Lite examples.**](https://vega.github.io/vega-lite/examples/)


    This dataset containers a collection of films and their performance
    metrics, including box office earnings, budgets, and audience ratings.
    In order to serve as a teaching resource, this dataset contains known data
    quality issues. However, it's still a very interesting dataset to play
    around with, as you'll see below.
    """

    st.space()

    """
    ## Part I: Ratings

    **How much do critics and viewers agree on ratings?** In the analysis
    below, a movie's Rotten Tomatoes Rating stands for what the professional
    critics think, while its IMDB Rating stands for what the general public
    thinks.

    Running a LOESS regression on the data, we find a pretty good correlation
    between the two variables, though with some prominent outliers (shown with
    :red[**red crosses**]).
    """


rating_df = (
    df.select(TITLE_COL, DIRECTOR_COL, IMDB_COL, RT_COL)
    .filter(HAS_RATINGS)
    .with_columns(
        delta=pl.col(IMDB_COL) / 10 - pl.col(RT_COL) / 100,
    )
)

rating_model_df = perform_loess_regression(
    rating_df, IMDB_COL, RT_COL, sigma_threshold=2
)

with st.container(horizontal=True):
    with st.container(width=GRID_WIDTH, height=GRID_HEIGHT):
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.subheader("Distribution of ratings from critics vs viewers")
            st.altair_chart(
                alt.Chart(rating_model_df)
                .mark_point(filled=True, size=MARK_SIZE, opacity=0.5)
                .encode(
                    alt.X(IMDB_COL, type="quantitative"),
                    alt.Y(RT_COL, type="quantitative"),
                    alt.Color("Status:N").legend(None),
                    alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                    tooltip=[TITLE_COL, DIRECTOR_COL, IMDB_COL, RT_COL, "Status"],
                ),
                height="stretch",
            )

        with cols[1]:
            st.space("medium")
            draw_histogram(rating_df, "IMDB Rating")
            draw_histogram(rating_df, "Rotten Tomatoes Rating")

    diff_df = rating_df.filter(
        pl.col(IMDB_COL).is_not_null() & pl.col(RT_COL).is_not_null()
    ).sort(by="delta", descending=True)

    help_text = (
        "This is calculated based on the delta between the IMDB and "
        "Rotten Tomatoes scores."
    )

    with st.container(width=GRID_WIDTH, height=GRID_HEIGHT):
        st.subheader(
            "Movies most beloved by viewers and disliked by critics",
            help=help_text,
        )

        st.dataframe(
            diff_df.select(pl.exclude("delta"))
            .head(20)
            .sort(by=IMDB_COL, descending=True),
            column_config=COLUMN_CONFIG,
            height="stretch",
        )

    with st.container(width=GRID_WIDTH, height=GRID_HEIGHT):
        st.subheader(
            "Movies most beloved by critics and disliked by viewers",
            help=help_text,
        )

        st.dataframe(
            diff_df.select(pl.exclude("delta"))
            .tail(20)
            .sort(by=RT_COL, descending=True),
            column_config=COLUMN_CONFIG,
            height="stretch",
        )


# -----------------------------------------------------------------------------
# Part 2

st.space("large")

with st.container(width=GRID_WIDTH):
    """
    ## Part II: Top and bottom-rated directors

    **Which directors are the most and least beloved?** Do critics and viewers
    agree on this at least?

    From a quick look at the shape of scatterplot below, even though the median
    scores for different directions are well-correlated between IMDB and Rotten
    Tomatoes, the list of top and bottom-rated directors for each are pretty
    different -- especially when considering the exact ordering.

    In the visualizations below, :rainbow[**colors**] and **shapes** are used to help
    more quickly identify specific directors across each chart.
    """

    min_movies = st.slider(
        "Only consider directors with at least this many movies",
        min_value=1,
        max_value=10,
        value=5,
    )


director_df = rating_df.filter(pl.col(DIRECTOR_COL).is_not_null()).with_columns(
    first_letter=pl.col(DIRECTOR_COL).str.head(1)
)

director_medians_df = (
    director_df.group_by(DIRECTOR_COL)
    .agg(
        **{
            "num_movies": pl.len(),
            "first_letter": pl.col("first_letter").first(),
            IMDB_COL: pl.col(IMDB_COL).median(),
            RT_COL: pl.col(RT_COL).median(),
        }
    )
    .filter(pl.col("num_movies") >= min_movies)
)

all_directors_list = director_medians_df.get_column(DIRECTOR_COL).to_list()

with st.container(height=GRID_HEIGHT, width=GRID_WIDTH):
    st.subheader("Distribution of ratings from critics vs viewers")
    st.altair_chart(
        alt.Chart(director_medians_df)
        .mark_point(filled=True, size=DIRECTOR_MARK_SIZE)
        .encode(
            alt.X(IMDB_COL, type="quantitative"),
            alt.Y(RT_COL, type="quantitative"),
            alt.Color(DIRECTOR_COL, type="nominal")
            .scale(domain=all_directors_list)
            .legend(None),
            alt.Shape(DIRECTOR_COL, type="nominal")
            .scale(domain=all_directors_list)
            .legend(None),
            tooltip=[DIRECTOR_COL, IMDB_COL, RT_COL, "num_movies"],
        ),
        height="stretch",
    )

with st.container(horizontal=True):
    for i in range(2):
        if i == 0:
            metric_name = IMDB_COL
            director_medians_df = director_medians_df.filter(
                pl.col(IMDB_COL).is_not_null()
            )
            x_domain = [0, 10]
        else:
            metric_name = RT_COL
            director_medians_df = director_medians_df.filter(
                pl.col(RT_COL).is_not_null()
            )
            x_domain = [0, 100]

        with st.container(width=GRID_WIDTH, border=True):
            top_directors_df = director_medians_df.sort(
                metric_name, descending=True
            ).head(10)

            top_directors_set = set(top_directors_df.get_column(DIRECTOR_COL).to_list())

            top_dir_df = director_df.filter(
                pl.col(DIRECTOR_COL).is_in(top_directors_set)
            )

            draw_director_median_chart(
                f"Top 10 directors by {metric_name}",
                data=top_dir_df,
                x_col=metric_name,
                y_col=DIRECTOR_COL,
                x_domain=x_domain,
                color_domain=all_directors_list,
            )

        with st.container(width=GRID_WIDTH, border=True):
            bottom_directors_df = director_medians_df.sort(metric_name).head(10)

            bottom_directors_set = set(
                bottom_directors_df.get_column(DIRECTOR_COL).to_list()
            )

            bottom_dir_df = director_df.filter(
                pl.col(DIRECTOR_COL).is_in(bottom_directors_set)
            )

            draw_director_median_chart(
                f"Bottom 10 directors by {metric_name}",
                data=bottom_dir_df,
                x_col=metric_name,
                y_col=DIRECTOR_COL,
                x_domain=x_domain,
                color_domain=all_directors_list,
            )


# -----------------------------------------------------------------------------
# Part 3

st.space("large")

with st.container(width=GRID_WIDTH):
    """
    ## Part III: Predictions

    **How well can we predict, say, the US DVD Sales of a movie given the movie's
    IMDB rating?** What if we user from the Rotten Tomatoes rating for the
    prediction instead? Use the knobs below to run a regression and find out.

    The :blue[**blue**] line is the model prediction, and outliers are shown as
    :red[**red crosses**].
    """

numeric_cols = [
    "US Gross",
    "Worldwide Gross",
    "US DVD Sales",
    "Production Budget",
    "IMDB Rating",
    "Rotten Tomatoes Rating",
]

with st.container(width=GRID_WIDTH):
    st.space()

    cols = st.columns(2)

    with cols[0]:
        x_col = st.selectbox(
            "X Axis (predictor)", options=numeric_cols, index=4
        )  # Default IMDB

    with cols[1]:
        y_col = st.selectbox(
            "Y Axis (target)", options=numeric_cols, index=2
        )  # Default DVD Sales

    sigma_val = st.slider(
        "Confidence interval (sigma)",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Determines the width of the confidence band. Points outside this band are outliers.",
    )

    regression_type = st.segmented_control(
        "Regression type",
        ["Linear regression", "LOESS regression"],
        default="Linear regression",
    )

    st.space()


if not x_col or not y_col:
    st.info("Please select columns to visualize.")
    st.stop()


if regression_type == "Linear regression":
    model_df = perform_linear_regression(df, x_col, y_col, sigma_val)
else:
    model_df = perform_loess_regression(df, x_col, y_col, sigma_val)

outliers = model_df.filter(pl.col("Status") == "Outlier").select(
    TITLE_COL, IMDB_COL, RT_COL
)
num_outliers = len(model_df.filter(pl.col("Status") == "Outlier"))

st.metric(
    "Number of outliers",
    num_outliers,
    help="These are films that the model cannot predict well.",
)

st.space()

with st.container(horizontal=True):
    with st.container(width=GRID_WIDTH, height=GRID_HEIGHT):
        base = alt.Chart(model_df).encode(
            alt.X(x_col, title=x_col, scale=alt.Scale(zero=False))
        )

        band = base.mark_area(opacity=0.1).encode(
            alt.Y("Lower Bound"),
            alt.Y2("Upper Bound"),
            tooltip=[
                alt.Tooltip("Lower Bound", format=",.0f"),
                alt.Tooltip("Upper Bound", format=",.0f"),
            ],
        )

        line = base.mark_line(size=3).encode(y="Predicted")

        points = base.mark_point(filled=True, size=MARK_SIZE).encode(
            alt.Y(y_col, title=y_col),
            alt.Color(
                "Status:N",
            ).legend(None),
            alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
            tooltip=[TITLE_COL, x_col, y_col],
        )

        final_chart = (band + points + line).configure_legend(orient="bottom")

        st.subheader(f"{y_col} by {x_col}")
        st.altair_chart(final_chart, width="stretch", height="stretch")

    with st.container(width=GRID_WIDTH, height=GRID_HEIGHT):
        st.subheader("List of outliers")
        st.dataframe(outliers, column_config=COLUMN_CONFIG, height="stretch")


# -----------------------------------------------------------------------------
# Part 4

st.space("large")

"""
## Part IV: Browse the full dataset
"""

with st.container(width=2 * GRID_WIDTH, height=GRID_HEIGHT):
    st.subheader("Full data")
    st.dataframe(
        df,
        height="stretch",
        column_config=COLUMN_CONFIG,
    )

st.space()

st.header(
    ":gray[:material/movie: :material/movie_info: :material/camera:]", anchor=False
)
