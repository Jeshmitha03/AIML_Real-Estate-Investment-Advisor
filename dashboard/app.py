# ============================================================
# REAL ESTATE INVESTMENT ADVISOR
# Streamlit Application
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

# app.py is inside:
# GUVI_3_project/dashboard/app.py
#
# Therefore:
# dirname(__file__)       -> dashboard
# dirname(dirname(...))   -> GUVI_3_project

PROJECT_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# 3. FILE PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "processed",
    "ml_ready_housing_prices.csv"
)

MODELS_PATH = os.path.join(
    PROJECT_PATH,
    "models"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    MODELS_PATH,
    "best_classification_model.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    MODELS_PATH,
    "best_regression_model.pkl"
)

MODEL_INFO_PATH = os.path.join(
    MODELS_PATH,
    "model_info.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_PATH,
    "feature_columns.pkl"
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Dataset": DATA_PATH,
    "Classification Model": CLASSIFICATION_MODEL_PATH,
    "Regression Model": REGRESSION_MODEL_PATH,
    "Model Information": MODEL_INFO_PATH,
    "Feature Columns": FEATURE_COLUMNS_PATH
}

missing_files = []

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):
        missing_files.append(
            f"{file_name}: {file_path}"
        )


if missing_files:

    st.error("❌ Required files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# 5. LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    model_info = joblib.load(
        MODEL_INFO_PATH
    )

    feature_columns = joblib.load(
        FEATURE_COLUMNS_PATH
    )

    return (
        classification_model,
        regression_model,
        model_info,
        feature_columns
    )


df = load_dataset()

(
    classification_model,
    regression_model,
    model_info,
    feature_columns
) = load_models()


# ============================================================
# 6. TITLE
# ============================================================

st.title("🏠 Real Estate Investment Advisor")

st.markdown(
    """
    ### Predict Property Investment Potential & Future Value

    This application uses Machine Learning to:

    - 🏠 Predict whether a property is a **Good Investment**
    - 💰 Estimate the property's **5-Year Future Value**
    - 📊 Analyze real-estate market trends
    - 📈 Understand property pricing patterns
    """
)


st.divider()


# ============================================================
# 7. FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(input_df):

    data = input_df.copy()

    # --------------------------------------------------------
    # Price per Square Feet
    # --------------------------------------------------------

    if (
        "Price_in_Lakhs" in data.columns
        and "Size_in_SqFt" in data.columns
    ):

        data["Calculated_Price_per_SqFt"] = (
            data["Price_in_Lakhs"] * 100000
        ) / data["Size_in_SqFt"]


    # --------------------------------------------------------
    # Property Age
    # --------------------------------------------------------

    if "Year_Built" in data.columns:

        data["Calculated_Age_of_Property"] = (
            2026 - data["Year_Built"]
        )


    # --------------------------------------------------------
    # Amenities Count
    # --------------------------------------------------------

    if "Amenities" in data.columns:

        data["Amenities_Count"] = (
            data["Amenities"]
            .fillna("")
            .astype(str)
            .apply(
                lambda x: len(
                    [
                        item
                        for item in x.split(",")
                        if item.strip()
                    ]
                )
            )
        )


    # --------------------------------------------------------
    # Amenity Density
    # --------------------------------------------------------

    if (
        "Amenities_Count" in data.columns
        and "Size_in_SqFt" in data.columns
    ):

        data["Amenity_Density_Score"] = (
            data["Amenities_Count"]
            / data["Size_in_SqFt"].replace(0, np.nan)
        )

        data["Amenity_Density_Score"] = (
            data["Amenity_Density_Score"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )


    # --------------------------------------------------------
    # Floor Ratio
    # --------------------------------------------------------

    if (
        "Floor_No" in data.columns
        and "Total_Floors" in data.columns
    ):

        data["Floor_Ratio"] = (
            data["Floor_No"]
            / data["Total_Floors"].replace(0, np.nan)
        )

        data["Floor_Ratio"] = (
            data["Floor_Ratio"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )


    # --------------------------------------------------------
    # Price Range
    # --------------------------------------------------------

    if "Price_in_Lakhs" in data.columns:

        def price_category(price):

            if price < 50:
                return "Below 50L"

            elif price < 100:
                return "50L-1Cr"

            elif price < 200:
                return "1Cr-2Cr"

            elif price < 500:
                return "2Cr-5Cr"

            else:
                return "Above 5Cr"

        data["Price_Range"] = (
            data["Price_in_Lakhs"]
            .apply(price_category)
        )


    # --------------------------------------------------------
    # Size Category
    # --------------------------------------------------------

    if "Size_in_SqFt" in data.columns:

        def size_category(size):

            if size < 600:
                return "Small"

            elif size < 1200:
                return "Medium"

            elif size < 2000:
                return "Large"

            elif size < 3000:
                return "Very Large"

            else:
                return "Luxury"

        data["Size_Category"] = (
            data["Size_in_SqFt"]
            .apply(size_category)
        )


    # --------------------------------------------------------
    # BHK Category
    # --------------------------------------------------------

    if "BHK" in data.columns:

        data["BHK_Category"] = (
            data["BHK"]
            .apply(
                lambda x:
                f"{int(x)} BHK"
                if x < 5
                else "5+ BHK"
            )
        )


    # --------------------------------------------------------
    # Parking
    # --------------------------------------------------------

    if "Parking_Space" in data.columns:

        data["Parking_Space"] = pd.to_numeric(
            data["Parking_Space"],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # Numeric missing values
    # --------------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        data[column] = (
            data[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )


    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
        )


    return data


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("🏠 Property Details")

st.sidebar.markdown(
    "Enter property information to get an investment prediction."
)


# ============================================================
# LOCATION
# ============================================================

state = st.sidebar.selectbox(
    "State",
    sorted(
        df["State"]
        .dropna()
        .astype(str)
        .unique()
    )
)


state_cities = sorted(
    df[
        df["State"].astype(str) == state
    ]["City"]
    .dropna()
    .astype(str)
    .unique()
)


city = st.sidebar.selectbox(
    "City",
    state_cities
)


city_localities = sorted(
    df[
        (
            df["State"].astype(str) == state
        )
        &
        (
            df["City"].astype(str) == city
        )
    ]["Locality"]
    .dropna()
    .astype(str)
    .unique()
)


locality = st.sidebar.selectbox(
    "Locality",
    city_localities
)


# ============================================================
# PROPERTY DETAILS
# ============================================================

property_types = sorted(
    df["Property_Type"]
    .dropna()
    .astype(str)
    .unique()
)

property_type = st.sidebar.selectbox(
    "Property Type",
    property_types
)


bhk = st.sidebar.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


size = st.sidebar.number_input(
    "Size (SqFt)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=50
)


price = st.sidebar.number_input(
    "Price (Lakhs)",
    min_value=1.0,
    max_value=5000.0,
    value=75.0,
    step=1.0
)


year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1950,
    max_value=2026,
    value=2015,
    step=1
)


# ============================================================
# PROPERTY CONDITION
# ============================================================

furnished_status = st.sidebar.selectbox(
    "Furnished Status",
    [
        "Unfurnished",
        "Semi-Furnished",
        "Fully Furnished"
    ]
)


floor_no = st.sidebar.number_input(
    "Floor Number",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)


total_floors = st.sidebar.number_input(
    "Total Floors",
    min_value=1,
    max_value=100,
    value=5,
    step=1
)


# ============================================================
# INFRASTRUCTURE
# ============================================================

nearby_schools = st.sidebar.number_input(
    "Nearby Schools",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)


nearby_hospitals = st.sidebar.number_input(
    "Nearby Hospitals",
    min_value=0,
    max_value=100,
    value=3,
    step=1
)


public_transport = st.sidebar.selectbox(
    "Public Transport Accessibility",
    [
        "Low",
        "Medium",
        "High"
    ]
)


parking = st.sidebar.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


security = st.sidebar.selectbox(
    "Security",
    [
        "No",
        "Yes"
    ]
)


# ============================================================
# OTHER FEATURES
# ============================================================

amenities = st.sidebar.text_input(
    "Amenities",
    value="Gym,Swimming Pool,Parking"
)


facing = st.sidebar.selectbox(
    "Facing",
    [
        "East",
        "West",
        "North",
        "South",
        "North-East",
        "North-West",
        "South-East",
        "South-West"
    ]
)


owner_type = st.sidebar.selectbox(
    "Owner Type",
    [
        "Owner",
        "Builder",
        "Dealer"
    ]
)


availability_status = st.sidebar.selectbox(
    "Availability Status",
    [
        "Ready to Move",
        "Under Construction"
    ]
)


# ============================================================
# 9. TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Prediction",
        "📊 Market Insights",
        "💰 Price Analysis",
        "🤖 Model Performance"
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.header("🔮 Property Investment Prediction")

    st.write(
        "Enter the property details from the sidebar "
        "and click the button below."
    )


    predict_button = st.button(
        "🚀 Predict Property",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "State": [state],
                "City": [city],
                "Locality": [locality],
                "Property_Type": [property_type],
                "BHK": [bhk],
                "Size_in_SqFt": [size],
                "Price_in_Lakhs": [price],
                "Year_Built": [year_built],
                "Furnished_Status": [furnished_status],
                "Floor_No": [floor_no],
                "Total_Floors": [total_floors],
                "Nearby_Schools": [nearby_schools],
                "Nearby_Hospitals": [nearby_hospitals],
                "Public_Transport_Accessibility": [
                    public_transport
                ],
                "Parking_Space": [parking],
                "Security": [security],
                "Amenities": [amenities],
                "Facing": [facing],
                "Owner_Type": [owner_type],
                "Availability_Status": [
                    availability_status
                ]
            }
        )


        try:

            # ------------------------------------------------
            # Feature engineering
            # ------------------------------------------------

            input_features = create_features(
                input_data
            )


            # ------------------------------------------------
            # Match training columns
            # ------------------------------------------------

            input_features = input_features.reindex(
                columns=feature_columns,
                fill_value=0
            )


            # ------------------------------------------------
            # Classification prediction
            # ------------------------------------------------

            classification_prediction = (
                classification_model.predict(
                    input_features
                )
            )[0]


            # ------------------------------------------------
            # Classification probability
            # ------------------------------------------------

            probability = None

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model.predict_proba(
                        input_features
                    )[0]
                )

                probability = (
                    np.max(probabilities) * 100
                )


            # ------------------------------------------------
            # Regression prediction
            # ------------------------------------------------

            future_price = (
                regression_model.predict(
                    input_features
                )
            )[0]


            # ------------------------------------------------
            # Calculate appreciation
            # ------------------------------------------------

            appreciation = (
                (
                    future_price - price
                )
                / price
            ) * 100


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.subheader(
                "🎯 Prediction Results"
            )


            col1, col2, col3 = st.columns(3)


            # ------------------------------------------------
            # Investment result
            # ------------------------------------------------

            with col1:

                if (
                    classification_prediction
                    in [1, "1", True, "Yes", "Good_Investment"]
                ):

                    st.success(
                        "🟢 Good Investment"
                    )

                else:

                    st.warning(
                        "🟡 Lower Investment Potential"
                    )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            with col2:

                if probability is not None:

                    st.metric(
                        "Prediction Confidence",
                        f"{probability:.2f}%"
                    )

                else:

                    st.metric(
                        "Prediction Confidence",
                        "N/A"
                    )


            # ------------------------------------------------
            # Future price
            # ------------------------------------------------

            with col3:

                st.metric(
                    "Estimated Value After 5 Years",
                    f"₹{future_price:,.2f} Lakhs"
                )


            st.divider()


            # =================================================
            # PRICE SUMMARY
            # =================================================

            st.subheader(
                "💰 Property Value Summary"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Current Price",
                    f"₹{price:,.2f} L"
                )


            with col2:

                st.metric(
                    "Future Value",
                    f"₹{future_price:,.2f} L"
                )


            with col3:

                st.metric(
                    "Expected Appreciation",
                    f"{appreciation:.2f}%"
                )


            with col4:

                calculated_price_sqft = (
                    price * 100000
                ) / size

                st.metric(
                    "Price / SqFt",
                    f"₹{calculated_price_sqft:,.0f}"
                )


            st.divider()


            # =================================================
            # PROPERTY SUMMARY
            # =================================================

            st.subheader(
                "🏠 Property Summary"
            )


            summary_col1, summary_col2 = st.columns(2)


            with summary_col1:

                st.write(
                    f"**State:** {state}"
                )

                st.write(
                    f"**City:** {city}"
                )

                st.write(
                    f"**Locality:** {locality}"
                )

                st.write(
                    f"**Property Type:** {property_type}"
                )

                st.write(
                    f"**BHK:** {bhk}"
                )


            with summary_col2:

                st.write(
                    f"**Size:** {size:,} SqFt"
                )

                st.write(
                    f"**Year Built:** {year_built}"
                )

                st.write(
                    f"**Furnished:** {furnished_status}"
                )

                st.write(
                    f"**Floor:** {floor_no}/{total_floors}"
                )

                st.write(
                    f"**Amenities:** {amenities}"
                )


            # =================================================
            # INVESTMENT RECOMMENDATION
            # =================================================

            st.divider()

            st.subheader(
                "💡 Investment Recommendation"
            )


            if (
                classification_prediction
                in [1, "1", True, "Yes", "Good_Investment"]
            ):

                st.success(
                    """
                    This property is predicted to have
                    good investment potential based on the
                    features provided.
                    """
                )

            else:

                st.warning(
                    """
                    This property has relatively lower
                    predicted investment potential.
                    Consider comparing it with other
                    properties before investing.
                    """
                )


            # =================================================
            # APPRECIATION CHART
            # =================================================

            st.subheader(
                "📈 Current vs Estimated Future Value"
            )


            chart_data = pd.DataFrame(
                {
                    "Stage": [
                        "Current",
                        "5-Year Estimate"
                    ],
                    "Value": [
                        price,
                        future_price
                    ]
                }
            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.barplot(
                data=chart_data,
                x="Stage",
                y="Value",
                ax=ax
            )

            ax.set_ylabel(
                "Property Value (Lakhs)"
            )

            ax.set_xlabel("")

            ax.set_title(
                "Current vs Estimated 5-Year Property Value"
            )

            st.pyplot(fig)


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# TAB 2 — MARKET INSIGHTS
# ============================================================

with tab2:

    st.header(
        "📊 Real Estate Market Insights"
    )


    # ========================================================
    # KPI SECTION
    # ========================================================

    total_properties = len(df)

    average_price = (
        df["Price_in_Lakhs"].mean()
    )

    average_size = (
        df["Size_in_SqFt"].mean()
    )


    if "Price_per_SqFt" in df.columns:

        average_price_sqft = (
            df["Price_per_SqFt"].mean()
        )

    else:

        average_price_sqft = (
            df["Price_in_Lakhs"] * 100000
            / df["Size_in_SqFt"]
        ).mean()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Properties",
            f"{total_properties:,}"
        )


    with col2:

        st.metric(
            "Average Price",
            f"₹{average_price:,.2f} L"
        )


    with col3:

        st.metric(
            "Average Size",
            f"{average_size:,.0f} SqFt"
        )


    with col4:

        st.metric(
            "Average Price/SqFt",
            f"₹{average_price_sqft:,.0f}"
        )


    st.divider()


    # ========================================================
    # PRICE DISTRIBUTION
    # ========================================================

    st.subheader(
        "💰 Property Price Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.histplot(
        df["Price_in_Lakhs"],
        bins=40,
        kde=True,
        ax=ax
    )


    ax.set_title(
        "Distribution of Property Prices"
    )

    ax.set_xlabel(
        "Price (Lakhs)"
    )

    ax.set_ylabel(
        "Number of Properties"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # CITY ANALYSIS
    # ========================================================

    st.subheader(
        "🏙️ Top Cities by Number of Properties"
    )


    city_counts = (
        df["City"]
        .value_counts()
        .head(10)
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    city_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Top 10 Cities by Property Count"
    )

    ax.set_xlabel("City")

    ax.set_ylabel(
        "Number of Properties"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # PROPERTY TYPE
    # ========================================================

    st.subheader(
        "🏠 Property Type Distribution"
    )


    property_type_counts = (
        df["Property_Type"]
        .value_counts()
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    property_type_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Properties by Property Type"
    )

    ax.set_xlabel(
        "Property Type"
    )

    ax.set_ylabel(
        "Count"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # BHK DISTRIBUTION
    # ========================================================

    st.subheader(
        "🛏️ BHK Distribution"
    )


    bhk_counts = (
        df["BHK"]
        .value_counts()
        .sort_index()
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    bhk_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Property Distribution by BHK"
    )

    ax.set_xlabel(
        "BHK"
    )

    ax.set_ylabel(
        "Number of Properties"
    )


    st.pyplot(fig)


# ============================================================
# TAB 3 — PRICE ANALYSIS
# ============================================================

with tab3:

    st.header(
        "💰 Property Price Analysis"
    )


    # ========================================================
    # AVG PRICE BY PROPERTY TYPE
    # ========================================================

    st.subheader(
        "Average Price by Property Type"
    )


    avg_price_type = (
        df.groupby("Property_Type")[
            "Price_in_Lakhs"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    avg_price_type.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Average Property Price by Type"
    )

    ax.set_xlabel(
        "Property Type"
    )

    ax.set_ylabel(
        "Average Price (Lakhs)"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # SIZE VS PRICE
    # ========================================================

    st.subheader(
        "📐 Property Size vs Price"
    )


    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    sns.scatterplot(
        data=sample_df,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        alpha=0.5,
        ax=ax
    )


    ax.set_title(
        "Property Size vs Price"
    )

    ax.set_xlabel(
        "Size (SqFt)"
    )

    ax.set_ylabel(
        "Price (Lakhs)"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # PRICE PER SQFT
    # ========================================================

    st.subheader(
        "₹ Price per Square Foot Distribution"
    )


    if "Price_per_SqFt" in df.columns:

        price_sqft = (
            pd.to_numeric(
                df["Price_per_SqFt"],
                errors="coerce"
            )
            .dropna()
        )

    else:

        price_sqft = (
            df["Price_in_Lakhs"] * 100000
            / df["Size_in_SqFt"]
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.histplot(
        price_sqft,
        bins=40,
        kde=True,
        ax=ax
    )


    ax.set_title(
        "Price per Square Foot Distribution"
    )

    ax.set_xlabel(
        "Price per SqFt (₹)"
    )

    ax.set_ylabel(
        "Number of Properties"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # PRICE RANGE
    # ========================================================

    st.subheader(
        "🏷️ Properties by Price Range"
    )


    if "Price_Range" in df.columns:

        price_range_counts = (
            df["Price_Range"]
            .value_counts()
        )

    else:

        def get_price_range(price):

            if price < 50:
                return "Below 50L"

            elif price < 100:
                return "50L-1Cr"

            elif price < 200:
                return "1Cr-2Cr"

            elif price < 500:
                return "2Cr-5Cr"

            else:
                return "Above 5Cr"


        price_range_counts = (
            df["Price_in_Lakhs"]
            .apply(get_price_range)
            .value_counts()
        )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    price_range_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Property Distribution by Price Range"
    )

    ax.set_xlabel(
        "Price Range"
    )

    ax.set_ylabel(
        "Number of Properties"
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    st.pyplot(fig)


    st.divider()


    # ========================================================
    # STATE PRICE ANALYSIS
    # ========================================================

    st.subheader(
        "📍 Average Price by State"
    )


    state_price = (
        df.groupby("State")[
            "Price_in_Lakhs"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    state_price.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Top States by Average Property Price"
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Average Price (Lakhs)"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    st.pyplot(fig)


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab4:

    st.header(
        "🤖 Machine Learning Model Performance"
    )


    # ========================================================
    # CLASSIFICATION MODEL
    # ========================================================

    st.subheader(
        "🎯 Classification Model"
    )


    classification_model_name = (
        model_info.get(
            "classification_model",
            "Not Available"
        )
    )


    classification_f1 = (
        model_info.get(
            "classification_f1",
            None
        )
    )


    classification_accuracy = (
        model_info.get(
            "classification_accuracy",
            None
        )
    )


    classification_precision = (
        model_info.get(
            "classification_precision",
            None
        )
    )


    classification_recall = (
        model_info.get(
            "classification_recall",
            None
        )
    )


    classification_roc_auc = (
        model_info.get(
            "classification_roc_auc",
            None
        )
    )


    st.write(
        f"**Best Classification Model:** "
        f"{classification_model_name}"
    )


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        if classification_accuracy is not None:

            st.metric(
                "Accuracy",
                f"{classification_accuracy:.2f}"
            )


    with col2:

        if classification_precision is not None:

            st.metric(
                "Precision",
                f"{classification_precision:.2f}"
            )


    with col3:

        if classification_recall is not None:

            st.metric(
                "Recall",
                f"{classification_recall:.2f}"
            )


    with col4:

        if classification_f1 is not None:

            st.metric(
                "F1 Score",
                f"{classification_f1:.2f}"
            )


    with col5:

        if classification_roc_auc is not None:

            st.metric(
                "ROC-AUC",
                f"{classification_roc_auc:.2f}"
            )


    st.divider()


    # ========================================================
    # REGRESSION MODEL
    # ========================================================

    st.subheader(
        "💰 Regression Model"
    )


    regression_model_name = (
        model_info.get(
            "regression_model",
            "Not Available"
        )
    )


    regression_mae = (
        model_info.get(
            "regression_mae",
            None
        )
    )


    regression_rmse = (
        model_info.get(
            "regression_rmse",
            None
        )
    )


    regression_r2 = (
        model_info.get(
            "regression_r2",
            None
        )
    )


    st.write(
        f"**Best Regression Model:** "
        f"{regression_model_name}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        if regression_mae is not None:

            st.metric(
                "MAE",
                f"{regression_mae:.4f}"
            )


    with col2:

        if regression_rmse is not None:

            st.metric(
                "RMSE",
                f"{regression_rmse:.4f}"
            )


    with col3:

        if regression_r2 is not None:

            st.metric(
                "R² Score",
                f"{regression_r2:.4f}"
            )


    st.divider()


    # ========================================================
    # ML WORKFLOW
    # ========================================================

    st.subheader(
        "🔄 Machine Learning Workflow"
    )


    workflow = [
        "1️⃣ Data Collection",
        "2️⃣ Data Cleaning",
        "3️⃣ Feature Engineering",
        "4️⃣ Exploratory Data Analysis",
        "5️⃣ Train/Test Split",
        "6️⃣ Data Preprocessing",
        "7️⃣ Model Training",
        "8️⃣ Model Evaluation",
        "9️⃣ Hyperparameter Tuning",
        "🔟 MLflow Experiment Tracking",
        "1️⃣1️⃣ Save Best Models",
        "1️⃣2️⃣ Streamlit Deployment"
    ]


    for step in workflow:

        st.write(step)


    st.divider()


    # ========================================================
    # PROJECT OBJECTIVES
    # ========================================================

    st.subheader(
        "🎯 Project Objectives"
    )


    st.markdown(
        """
        **Classification Objective**

        Predict whether a property is likely to be a
        **Good Investment**.

        **Regression Objective**

        Estimate the property's value after five years.

        **Business Objective**

        Help property buyers and investors compare
        properties using data-driven insights.
        """
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 Real Estate Investment Advisor | "
    "Machine Learning Project"
)