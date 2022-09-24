import pickle
import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Used car picker",
    page_icon="🚗",
    menu_items={
        "About": "Used Car Picker\n"
        "\nApp contact: [Siddhant Sadangi](mailto:siddhant.sadangi@gmail.com)",
        "Report a Bug": "https://github.com/SiddhantSadangi/carpicker/issues/new",
        "Get help": None,
    },
    layout="wide",
)

# ---------- HEADER ----------
st.title("Welcome to the Used Car Picker app!")
st.subheader("Get the car with the highest resale profit potential!")
st.caption("Enter your preferences on the left to find which car might earn you the most profit (or least loss) after resale.")

# ---------- FUNCTIONS ----------
def _reset():
    st.session_state["price_range"] = [data_clipped_df.price.min(), data_clipped_df.price.max()]
    st.session_state["duration"] = 0
    st.session_state["usage"] = 0
    st.session_state["fuel_filter"] = st.session_state[
        "paint_filter"
    ] = st.session_state["type_filter"] = []
    st.session_state["feature1"] = st.session_state["feature2"] = st.session_state[
        "feature3"
    ] = st.session_state["feature4"] = st.session_state["feature5"] = st.session_state[
        "feature6"
    ] = st.session_state[
        "feature7"
    ] = st.session_state[
        "feature8"
    ] = False


# ---------- LOADING DATA ----------
data_clipped_df = pd.read_csv("data_clipped.csv", index_col="index")
data_clipped_df["price"] = data_clipped_df["price"].astype(int)

with open("regressor.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("params.json", encoding='utf8') as f:
    params = json.load(f)

numerical_cols = params.get("numerical_cols")
imp_features = params.get("imp_features")

# -------------- INPUT --------------
# ------------- SIDEBAR -------------
with st.sidebar:

    duration = st.number_input(
        label="How long will you keep the car for (in years)?",
        min_value=0.0,
        value=0.0,
        step=0.5,
        key="duration",
    )

    usage = st.number_input(
        label="How much will you drive the car (in miles)?",
        min_value=0,
        value=0,
        step=1000,
        key="usage",
    )

## -- ADDING USAGE AND DURATION --
future_space_df = data_clipped_df.copy()
future_space_df["age"] = data_clipped_df["age"] + duration
future_space_df["mileage"] = data_clipped_df["mileage"] + usage

# ---------- OPERATIONS ----------

## - ENCODING CATEGORICAL FEATURES -
encoded_space_df = pd.get_dummies(future_space_df, drop_first=True)

## CALCULATING RESALE VALUE AND PROFIT/LOSS
preds = np.round(regressor.predict(encoded_space_df[imp_features]))

data_clipped_df["resale_price"] = list(preds)
data_clipped_df["resale_price"] = data_clipped_df["resale_price"].astype(int)
data_clipped_df["profit"] = data_clipped_df.resale_price - data_clipped_df.price

# --------- ADDING OPTIONAL FILTERS ---------
with st.sidebar:

    price_range = st.slider(
        label="Budget",
        min_value=int(data_clipped_df.price.min()),
        max_value=int(data_clipped_df.price.max()),
        value=[int(data_clipped_df.price.min()), int(data_clipped_df.price.max())],
        step=100,
        key="price_range",
    )
    space_df = data_clipped_df[
        data_clipped_df["price"].between(price_range[0], price_range[1])
    ]

    if fuel_filter := st.multiselect(
        label="Preferred fuel",
        options=data_clipped_df["fuel"].unique(),
        key="fuel_filter",
    ):
        space_df = space_df[space_df["fuel"].isin(fuel_filter)]

    if paint_filter := st.multiselect(
        label="Preferred paint color",
        options=data_clipped_df["paint_color"].unique(),
        key="paint_filter",
    ):
        space_df = space_df[space_df["paint_color"].isin(paint_filter)]

    if type_filter := st.multiselect(
        label="Preferred car type",
        options=data_clipped_df["car_type"].unique(),
        key="type_filter",
    ):
        space_df = space_df[space_df["car_type"].isin(type_filter)]

    st.subheader("Other features needed")

    col1, col2, col3 = st.columns(3)

    if feature1 := col1.checkbox(label="Feature1", key="feature1"):
        space_df = space_df[space_df["feature_1"] == feature1]

    if feature2 := col2.checkbox(label="Feature2", key="feature2"):
        space_df = space_df[space_df["feature_2"] == feature2]

    if feature3 := col3.checkbox(label="Feature3", key="feature3"):
        space_df = space_df[space_df["feature_3"] == feature3]

    if feature4 := col1.checkbox(label="Feature4", key="feature4"):
        space_df = space_df[space_df["feature_4"] == feature4]

    if feature5 := col2.checkbox(label="Feature5", key="feature5"):
        space_df = space_df[space_df["feature_5"] == feature5]

    if feature6 := col3.checkbox(label="Feature6", key="feature6"):
        space_df = space_df[space_df["feature_6"] == feature6]

    if feature7 := col1.checkbox(label="Feature7", key="feature7"):
        space_df = space_df[space_df["feature_7"] == feature7]

    if feature8 := col2.checkbox(label="Feature8", key="feature8"):
        space_df = space_df[space_df["feature_8"] == feature8]

    col1, col2_3 = st.columns((1,2))
    if col1.button("Reset All", on_click=_reset):
        col2_3.success("App reset!")

## ----- COMMUNICATING RESULTS -----
st.subheader(f"Cars available: {len(space_df)}")

try:
    best_purchase = space_df.sort_values("profit", ascending=False).iloc[0]
except IndexError:
    st.subheader("There are no cars available that meet your preferences 😔")
else:
    st.dataframe(space_df.sort_values("profit", ascending=False).reset_index(drop=True), use_container_width=True)
    st.subheader("Best option")
    st.write(
        f"""<font size=5>
        Based on your preferences and usage, you should purchase a <b><u>{best_purchase.age: .1f} years old BMW {best_purchase.model_key} for around ${best_purchase.price:,}</u></b>.<br>
        This can potentially sell for around <b>${best_purchase.resale_price:,}</b>, leaving you with a 
        <b><font color={'red' if best_purchase.profit < 0 else 'green'}> {'loss' if best_purchase.profit < 0 else 'profit'} </font> of ${abs(best_purchase.profit):,}</b>.
        </font>""",
        unsafe_allow_html=True,
    )
