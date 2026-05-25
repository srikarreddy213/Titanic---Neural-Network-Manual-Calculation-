import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Titanic Neural Network", layout="wide")

st.title("🚢 Titanic Survival Prediction using Neural Network")

st.markdown("""
This project demonstrates:

- Forward Propagation
- Error Calculation
- Backpropagation
- Weight Updates

using one Titanic dataset record.
""")

# -------------------------
# INPUT VALUES
# -------------------------

x1 = 0.20
x2 = 0.24
x3 = 0.80

target = 1
learning_rate = 0.1

# Initial Weights

w1 = 0.11
w2 = 0.14
w3 = 0.17

w4 = 0.21
w5 = 0.24
w6 = 0.27

# Biases

bh1 = 0.1
bh2 = 0.1

# Hidden → Output

w7 = 0.31
w8 = 0.34

bo = 0.1

# -------------------------
# SIGMOID FUNCTION
# -------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# -------------------------
# TASK 1
# -------------------------

st.header("Task 1 : Net Input")

net_h1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + bh1
net_h2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + bh2

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hidden Neuron h1")
    st.success(f"Net Input h1 = {net_h1:.4f}")

with col2:
    st.subheader("Hidden Neuron h2")
    st.success(f"Net Input h2 = {net_h2:.4f}")

# -------------------------
# TASK 2
# -------------------------

st.header("Task 2 : Sigmoid Activation")

h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Output h1")
    st.info(f"h1 = {h1:.4f}")

with col4:
    st.subheader("Output h2")
    st.info(f"h2 = {h2:.4f}")

# -------------------------
# TASK 3
# -------------------------

st.header("Task 3 : Output Layer")

net_o = (h1 * w7) + (h2 * w8) + bo
output = sigmoid(net_o)

st.write(f"Net Output = {net_o:.4f}")
st.success(f"Final Prediction = {output:.4f}")

# -------------------------
# TASK 4
# -------------------------

st.header("Task 4 : Mean Squared Error")

mse = 0.5 * ((target - output) ** 2)

st.error(f"MSE = {mse:.4f}")

# -------------------------
# TASK 5
# -------------------------

st.header("Task 5 : Backpropagation")

delta_o = (target - output) * output * (1 - output)

delta_h1 = h1 * (1 - h1) * (delta_o * w7)
delta_h2 = h2 * (1 - h2) * (delta_o * w8)

col5, col6, col7 = st.columns(3)

with col5:
    st.write(f"Output Gradient = {delta_o:.4f}")

with col6:
    st.write(f"Hidden Gradient h1 = {delta_h1:.4f}")

with col7:
    st.write(f"Hidden Gradient h2 = {delta_h2:.4f}")

# -------------------------
# TASK 6
# -------------------------

st.header("Task 6 : Weight Updates")

# Hidden → Output Updates

new_w7 = w7 + (learning_rate * delta_o * h1)
new_w8 = w8 + (learning_rate * delta_o * h2)

# Output Bias

new_bo = bo + (learning_rate * delta_o)

# Input → Hidden Updates

new_w1 = w1 + (learning_rate * delta_h1 * x1)
new_w2 = w2 + (learning_rate * delta_h1 * x2)
new_w3 = w3 + (learning_rate * delta_h1 * x3)

new_w4 = w4 + (learning_rate * delta_h2 * x1)
new_w5 = w5 + (learning_rate * delta_h2 * x2)
new_w6 = w6 + (learning_rate * delta_h2 * x3)

# Bias Updates

new_bh1 = bh1 + (learning_rate * delta_h1)
new_bh2 = bh2 + (learning_rate * delta_h2)

# -------------------------
# TABLES
# -------------------------

weights_df = pd.DataFrame({
    "Connection": [
        "x1 → h1",
        "x2 → h1",
        "x3 → h1",
        "x1 → h2",
        "x2 → h2",
        "x3 → h2",
        "h1 → o1",
        "h2 → o1"
    ],
    "Updated Weight": [
        round(new_w1, 4),
        round(new_w2, 4),
        round(new_w3, 4),
        round(new_w4, 4),
        round(new_w5, 4),
        round(new_w6, 4),
        round(new_w7, 4),
        round(new_w8, 4)
    ]
})

st.subheader("Updated Weights")
st.table(weights_df)

bias_df = pd.DataFrame({
    "Bias": ["bh1", "bh2", "bo"],
    "Updated Value": [
        round(new_bh1, 4),
        round(new_bh2, 4),
        round(new_bo, 4)
    ]
})

st.subheader("Updated Biases")
st.table(bias_df)

# -------------------------
# SUMMARY
# -------------------------

st.header("Final Summary")

summary_df = pd.DataFrame({
    "Step": [
        "Hidden Output h1",
        "Hidden Output h2",
        "Final Prediction",
        "MSE Error",
        "Output Gradient",
        "Hidden Gradient h1",
        "Hidden Gradient h2"
    ],
    "Value": [
        round(h1, 4),
        round(h2, 4),
        round(output, 4),
        round(mse, 4),
        round(delta_o, 4),
        round(delta_h1, 4),
        round(delta_h2, 4)
    ]
})

st.dataframe(summary_df)

st.success("Neural Network Calculation Completed Successfully")
