
import os

import joblib
import pandas as pd
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. FILE PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "diamond_knn_pipeline.pkl",
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "model_comparison.csv",
)


# ==================================================
# 3. CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fc;
        color: #243b53;
    }

    [data-testid="stHeader"] {
        background-color: #f8f9fc;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    label,
    h1, h2, h3, h4 {
        color: #243b53 !important;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        color: #243b53 !important;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #243b53 !important;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 4. LOAD KNN MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.isfile(MODEL_PATH):
    st.error(
        "KNN model not found. Please check "
        "models/diamond_knn_pipeline.pkl"
    )
    st.stop()

try:
    model = load_model()

except Exception as e:
    st.error(f"Error loading KNN model: {e}")
    st.stop()


# ==================================================
# 5. LOAD MODEL COMPARISON DATA
# ==================================================

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


# ==================================================
# 6. HEADER
# ==================================================

st.title("💎 Diamond Price Prediction System")

st.markdown(
    """
    ### Welcome to the Diamond Price Prediction Dashboard

    Predict diamond prices using Machine Learning.

    Explore diamond characteristics, obtain instant price
    predictions and compare four regression algorithms.
    """
)

st.divider()


# ==================================================
# 7. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("💎 Diamond ML App")

    st.subheader("About the Project")

    st.write(
        """
        This application predicts diamond prices based
        on physical and quality characteristics.

        **Machine Learning:** Regression

        **Prediction Algorithm:** KNN Regressor

        **Algorithms Compared:**
        - Linear Regression
        - Decision Tree
        - Random Forest
        - KNN

        **Interface:** Streamlit
        """
    )

    st.divider()

    st.subheader("Project Information")

    st.write("Dataset: Diamonds Dataset")
    st.write("Training Split: 75%")
    st.write("Testing Split: 25%")

    st.divider()

    st.caption(
        "Diamond Price Prediction | Data Science Project"
    )


# ==================================================
# 8. DASHBOARD TABS
# ==================================================

tab1, tab2 = st.tabs(
    [
        "💍 Price Prediction",
        "📊 Model Comparison",
    ]
)


# ==================================================
# TAB 1: PRICE PREDICTION
# ==================================================

with tab1:

    st.header("💍 Enter Diamond Details")

    st.write(
        "Provide the following information "
        "to predict the diamond price."
    )

    with st.form("diamond_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("💎 Diamond Quality")

            carat = st.number_input(
                "Carat",
                min_value=0.20,
                max_value=5.00,
                value=0.50,
                step=0.01,
            )

            cut = st.selectbox(
                "Cut",
                [
                    "Ideal",
                    "Premium",
                    "Very Good",
                    "Good",
                    "Fair",
                ],
            )

            color = st.selectbox(
                "Color",
                ["D", "E", "F", "G", "H", "I", "J"],
            )

            clarity = st.selectbox(
                "Clarity",
                [
                    "IF",
                    "VVS1",
                    "VVS2",
                    "VS1",
                    "VS2",
                    "SI1",
                    "SI2",
                    "I1",
                ],
            )

        with col2:

            st.subheader("📏 Diamond Measurements")

            depth = st.number_input(
                "Depth (%)",
                min_value=40.0,
                max_value=80.0,
                value=61.5,
                step=0.1,
            )

            table = st.number_input(
                "Table (%)",
                min_value=40.0,
                max_value=80.0,
                value=55.0,
                step=0.1,
            )

            x = st.number_input(
                "Length (x) in mm",
                min_value=0.1,
                max_value=11.0,
                value=5.00,
                step=0.01,
            )

            y = st.number_input(
                "Width (y) in mm",
                min_value=0.1,
                max_value=11.0,
                value=5.05,
                step=0.01,
            )

            z = st.number_input(
                "Height (z) in mm",
                min_value=0.1,
                max_value=7.0,
                value=3.10,
                step=0.01,
            )

        st.divider()

        predict_button = st.form_submit_button(
            "💎 Predict Diamond Price",
            type="primary",
            use_container_width=True,
        )

    # ==============================================
    # PREDICTION LOGIC
    # ==============================================

    if predict_button:

        input_data = pd.DataFrame(
            {
                "carat": [carat],
                "cut": [cut],
                "color": [color],
                "clarity": [clarity],
                "depth": [depth],
                "table": [table],
                "x": [x],
                "y": [y],
                "z": [z],
            }
        )

        try:

            # Check that measurements are reasonable

            if x <= 0 or y <= 0 or z <= 0:
                st.error(
                    "Diamond measurements must be greater than zero."
                )

            elif x < y * 0.5 or x > y * 2:
                st.warning(
                    "Please verify the diamond length and width."
                )

            else:

                # Generate prediction

                prediction = model.predict(input_data)

                predicted_price = float(prediction[0])

                if predicted_price < 0:
                    st.error(
                        "The model generated an invalid negative price."
                    )

                else:

                    st.divider()

                    st.success(
                        "🎉 Diamond Price Prediction Completed!"
                    )

                    st.header("📊 Prediction Result")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.metric(
                            "Estimated Diamond Price",
                            f"${predicted_price:,.2f}",
                        )

                    with result_col2:

                        st.metric(
                            "Diamond Carat",
                            f"{carat:.2f}",
                        )

                    st.info(
                        "Prediction generated using the "
                        "KNN Regressor model."
                    )

                    st.subheader("💎 Diamond Details")

                    st.dataframe(
                        input_data,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Download prediction details

                    download_data = input_data.copy()

                    download_data["Predicted Price"] = (
                        predicted_price
                    )

                    st.download_button(
                        label="📥 Download Prediction Details",
                        data=download_data.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="diamond_prediction_details.csv",
                        mime="text/csv",
                    )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ==================================================
# TAB 2: MODEL COMPARISON DASHBOARD
# ==================================================

with tab2:

    st.header("📊 Machine Learning Model Comparison")

    st.write(
        """
        Compare four regression algorithms using
        MAE, RMSE and R² Score.
        """
    )

    if os.path.isfile(COMPARISON_PATH):

        try:

            comparison_df = load_comparison()

            required_columns = [
                "Model",
                "MAE",
                "RMSE",
                "R2 Score",
            ]

            # Check required columns

            missing_columns = [
                column
                for column in required_columns
                if column not in comparison_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing columns in comparison CSV: "
                    f"{missing_columns}"
                )

            elif comparison_df.empty:

                st.warning(
                    "Comparison CSV is empty."
                )

            else:

                # Convert metrics into numeric values

                for column in [
                    "MAE",
                    "RMSE",
                    "R2 Score",
                ]:

                    comparison_df[column] = pd.to_numeric(
                        comparison_df[column],
                        errors="coerce",
                    )

                comparison_df = comparison_df.dropna(
                    subset=[
                        "Model",
                        "MAE",
                        "RMSE",
                        "R2 Score",
                    ]
                )

                if comparison_df.empty:

                    st.error(
                        "No valid model metrics found."
                    )

                else:

                    # Identify best model using lowest RMSE

                    comparison_df = comparison_df.sort_values(
                        by="RMSE",
                        ascending=True,
                    ).reset_index(drop=True)

                    best_model = comparison_df.iloc[0]

                    st.success(
                        f"🏆 Best Performing Model: "
                        f"{best_model['Model']}"
                    )

                    st.divider()

                    # ==================================
                    # BEST MODEL PERFORMANCE
                    # ==================================

                    st.subheader(
                        "🏆 Best Model Performance"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "MAE",
                            f"{best_model['MAE']:,.2f}",
                        )

                    with col2:

                        st.metric(
                            "RMSE",
                            f"{best_model['RMSE']:,.2f}",
                        )

                    with col3:

                        st.metric(
                            "R² Score",
                            f"{best_model['R2 Score'] * 100:.2f}%",
                        )

                    st.divider()

                    # ==================================
                    # COMPARISON TABLE
                    # ==================================

                    st.subheader(
                        "📋 Complete Model Comparison"
                    )

                    display_df = comparison_df.copy()

                    display_df["R2 Score"] = (
                        display_df["R2 Score"] * 100
                    )

                    display_df = display_df.rename(
                        columns={
                            "R2 Score": "R² Score (%)"
                        }
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.divider()

                    # ==================================
                    # R2 SCORE CHART
                    # ==================================

                    st.subheader(
                        "📈 R² Score Comparison"
                    )

                    r2_chart = (
                        comparison_df
                        .set_index("Model")[["R2 Score"]]
                        * 100
                    )

                    st.bar_chart(
                        r2_chart,
                        y="R2 Score",
                    )

                    st.caption(
                        "Higher R² Score is better."
                    )

                    st.divider()

                    # ==================================
                    # MAE CHART
                    # ==================================

                    st.subheader(
                        "📉 MAE Comparison"
                    )

                    mae_chart = comparison_df.set_index(
                        "Model"
                    )[["MAE"]]

                    st.bar_chart(
                        mae_chart,
                        y="MAE",
                    )

                    st.caption(
                        "Lower MAE is better."
                    )

                    st.divider()

                    # ==================================
                    # RMSE CHART
                    # ==================================

                    st.subheader(
                        "📉 RMSE Comparison"
                    )

                    rmse_chart = comparison_df.set_index(
                        "Model"
                    )[["RMSE"]]

                    st.bar_chart(
                        rmse_chart,
                        y="RMSE",
                    )

                    st.caption(
                        "Lower RMSE is better."
                    )

                    st.divider()

                    # ==================================
                    # PERFORMANCE ANALYSIS
                    # ==================================

                    st.subheader(
                        "💡 Model Performance Analysis"
                    )

                    st.write(
                        f"""
                        - **Best Model:** {best_model['Model']}
                        - **MAE:** {best_model['MAE']:.2f}
                        - **RMSE:** {best_model['RMSE']:.2f}
                        - **R² Score:** {best_model['R2 Score']:.4f}
                        """
                    )

                    st.info(
                        """
                        The KNN Regressor is currently used for
                        price prediction.

                        The comparison dashboard identifies the
                        model with the lowest RMSE.

                        The selected model can be used for final
                        deployment after validating its performance.
                        """
                    )

                    st.divider()

                    # ==================================
                    # DOWNLOAD COMPARISON REPORT
                    # ==================================

                    st.subheader(
                        "📥 Download Comparison Report"
                    )

                    csv_data = comparison_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Comparison Report",
                        data=csv_data,
                        file_name="diamond_model_comparison.csv",
                        mime="text/csv",
                    )

        except Exception as e:

            st.error(
                f"Error loading model comparison: {e}"
            )

    else:

        st.warning(
            "Model comparison file not found. "
            "Please check model_comparison.csv "
            "in your project folder."
        )


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.caption(
    """
    Developed using Python, Pandas, Scikit-learn,
    Joblib and Streamlit.
    """
)

st.caption(
    "Diamond Price Prediction | "
    "End-to-End Machine Learning Project"
)
