import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# PAGE TITLE
st.title("KNN Regression App")

st.write("Predict Tip Amount using Total Bill")

# LOAD DATASET
df = sns.load_dataset("tips")

# SHOW DATASET

if st.checkbox("Show Dataset"):
    st.write(df.head())


# FEATURES & TARGET
X = df[["total_bill"]]
y = df["tip"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# HYPERPARAMETER

st.subheader("Select Hyperparameter")

k = st.slider("Number of Neighbors (K)", 1, 20, 5)

# MODEL TRAINING

model = KNeighborsRegressor(n_neighbors=k)

model.fit(X_train, y_train)

# PREDICTIONS

y_pred = model.predict(X_test)

# MODEL PERFORMANCE

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"MSE: {mse:.2f}")

st.write(f"R² Score: {r2:.2f}")

# USER INPUT

st.subheader("Predict Tip")

bill = st.slider(
    "Total Bill Amount",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    20.0,
)

# PREDICTION

input_data = scaler.transform([[bill]])

prediction = model.predict(input_data)

st.success(f"Predicted Tip: ${prediction[0]:.2f}")

# VISUALIZATION

st.subheader("KNN Regression Graph")

fig, ax = plt.subplots()

# actual data
ax.scatter(df["total_bill"], df["tip"])

# smooth prediction line
X_range = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100).reshape(
    -1, 1
)

X_range_scaled = scaler.transform(X_range)

y_range_pred = model.predict(X_range_scaled)

ax.plot(X_range, y_range_pred, color="red")

ax.set_xlabel("Total Bill")

ax.set_ylabel("Tip")

ax.set_title("Total Bill vs Tip")

st.pyplot(fig)
