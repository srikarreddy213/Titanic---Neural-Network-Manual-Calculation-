

# 1. app.py

```python
import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #020617);
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 30px;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white;
}

.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.big-font {
    font-size: 26px;
    font-weight: bold;
}

.metric {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------

st.markdown("""
<div style='text-align:center;padding:20px;'>
<h1 style='font-size:60px;'>🚢 Titanic Survival Prediction System</h1>
<h3 style='color:#38bdf8;'>Deep Learning Based Passenger Survival Prediction</h3>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🧍 Passenger Information")

pclass = st.sidebar.slider("Passenger Class (Normalized)", 0.0, 1.0, 0.20)
age = st.sidebar.slider("Age (Normalized)", 0.0, 1.0, 0.24)
fare = st.sidebar.slider("Fare (Normalized)", 0.0, 1.0, 0.80)

actual_output = st.sidebar.selectbox("Actual Survival", [0,1])

# ---------------------------------------------------
# WEIGHTS
# ---------------------------------------------------

w1 = 0.11
w2 = 0.14
w3 = 0.17

w4 = 0.21
w5 = 0.24
w6 = 0.27

bh1 = 0.1
bh2 = 0.1

w7 = 0.31
w8 = 0.34

bo = 0.1

learning_rate = 0.1

# ---------------------------------------------------
# SIGMOID
# ---------------------------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("🚀 Predict Survival"):

    # ---------------- FORWARD PROPAGATION ----------------

    net_h1 = (pclass * w1) + (age * w2) + (fare * w3) + bh1
    net_h2 = (pclass * w4) + (age * w5) + (fare * w6) + bh2

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    net_o = (h1 * w7) + (h2 * w8) + bo

    prediction = sigmoid(net_o)

    # ---------------- RESULTS ----------------

    st.markdown("## 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric'>
        <h3>Hidden Output h1</h3>
        <h1>{h1:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric'>
        <h3>Hidden Output h2</h3>
        <h1>{h2:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric'>
        <h3>Prediction</h3>
        <h1>{prediction:.4f}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------------- SURVIVAL STATUS ----------------

    if prediction >= 0.5:
        st.success("✅ Passenger Predicted to SURVIVE")
    else:
        st.error("❌ Passenger Predicted to NOT SURVIVE")

    # ---------------- ERROR ----------------

    mse = 0.5 * ((actual_output - prediction) ** 2)

    st.markdown("## 📉 Error Calculation")

    st.info(f"Mean Squared Error = {mse:.6f}")

    # ---------------- BACKPROPAGATION ----------------

    delta_o = (actual_output - prediction) * prediction * (1 - prediction)

    delta_h1 = h1 * (1 - h1) * (delta_o * w7)
    delta_h2 = h2 * (1 - h2) * (delta_o * w8)

    st.markdown("## 🔁 Backpropagation")

    bp_df = pd.DataFrame({
        "Gradient": [
            "Output Gradient",
            "Hidden Gradient h1",
            "Hidden Gradient h2"
        ],
        "Value": [
            round(delta_o,6),
            round(delta_h1,6),
            round(delta_h2,6)
        ]
    })

    st.dataframe(bp_df, use_container_width=True)

    # ---------------- WEIGHT UPDATES ----------------

    new_w7 = w7 + (learning_rate * delta_o * h1)
    new_w8 = w8 + (learning_rate * delta_o * h2)

    new_bo = bo + (learning_rate * delta_o)

    new_w1 = w1 + (learning_rate * delta_h1 * pclass)
    new_w2 = w2 + (learning_rate * delta_h1 * age)
    new_w3 = w3 + (learning_rate * delta_h1 * fare)

    new_w4 = w4 + (learning_rate * delta_h2 * pclass)
    new_w5 = w5 + (learning_rate * delta_h2 * age)
    new_w6 = w6 + (learning_rate * delta_h2 * fare)

    st.markdown("## ⚡ Updated Weights")

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
            round(new_w1,6),
            round(new_w2,6),
            round(new_w3,6),
            round(new_w4,6),
            round(new_w5,6),
            round(new_w6,6),
            round(new_w7,6),
            round(new_w8,6)
        ]
    })

    st.dataframe(weights_df, use_container_width=True)

    # ---------------- BIAS ----------------

    st.markdown("## 🧠 Updated Biases")

    bias_df = pd.DataFrame({
        "Bias": ["bh1","bh2","bo"],
        "Updated Value": [
            round(bh1 + learning_rate * delta_h1,6),
            round(bh2 + learning_rate * delta_h2,6),
            round(new_bo,6)
        ]
    })

    st.dataframe(bias_df, use_container_width=True)

    # ---------------- FINAL SUMMARY ----------------

    st.markdown("## 📌 Final Summary")

    summary_df = pd.DataFrame({
        "Step": [
            "Prediction",
            "MSE Error",
            "Output Gradient",
            "Hidden Gradient h1",
            "Hidden Gradient h2"
        ],
        "Value": [
            round(prediction,6),
            round(mse,6),
            round(delta_o,6),
            round(delta_h1,6),
            round(delta_h2,6)
        ]
    })

    st.table(summary_df)

st.divider()

st.markdown("""
<div style='text-align:center;'>
<h2>Developed using Streamlit + Artificial Neural Network</h2>
<p style='color:gray;'>Interactive Deep Learning Mini Project</p>
</div>
""", unsafe_allow_html=True)
```

---

# 2. requirements.txt

```txt
streamlit
pandas
```

---

# 3. README.md

````md
# Titanic Neural Network Manual Calculation

This project demonstrates:

- Forward Propagation
- Sigmoid Activation
- Mean Squared Error
- Backpropagation
- Weight Updates

using a simple neural network in Streamlit.

---

# Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

# Deploy on Streamlit Cloud

1. Upload project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub Repository
4. Select app.py
5. Deploy

---

# Technologies Used

* Python
* Streamlit
* Pandas

````

---

# 4. data.csv

```csv
Pclass,Age,Fare,Survived
1,24,120,1
````

---

# GitHub Upload Steps

```bash
git init
git add .
git commit -m "Initial Commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_LINK
git push -u origin main
```

---

# Streamlit Deployment

Website:

[https://streamlit.io/cloud](https://streamlit.io/cloud)

1. Sign in using GitHub
2. Click New App
3. Select Repository
4. Select app.py
5. Click Deploy

Your website will be live in 2 minutes.
