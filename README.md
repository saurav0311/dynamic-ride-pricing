# 🚕 Dynamic Ride-Sharing Pricing using Wide & Deep Neural Network

An **end-to-end Deep Learning regression project** that predicts **optimal ride fares** using tabular data, inspired by real-world ride-sharing systems like **Uber/Ola**.

---

## 📌 Problem Statement
Static pricing fails under dynamic conditions such as:
- Peak hours
- Long-distance trips
- Weekend demand

The goal is to build a **dynamic pricing model** that produces **stable and realistic fare predictions**.

---

## 🧠 Approach
- **Baseline Model:** Linear Regression  
- **Advanced Model:** Wide & Deep Neural Network  

### 🔹 Wide (Linear) Part
- Peak hour
- Weekend
- Passenger count  

### 🔹 Deep (Neural Network) Part
- Distance (Haversine)
- Hour of day
- Day of week  

The **Wide part memorizes simple rules**, while the **Deep part learns complex feature interactions**.

---

## 📊 Dataset
- **NYC Taxi Fare Prediction Dataset (Kaggle)**
- Cleaned invalid fares, coordinates, and outliers
- Engineered distance using **Haversine formula**

---

## 📈 Results

| Model | MAE | RMSE |
|------|-----|------|
| Linear Regression | ~2.34 | ~5.46 |
| Wide & Deep NN | ~2.31 | ~5.25 |

✅ Reduced extreme pricing errors  
✅ Better handling of complex trips (lower RMSE)

---

## 🖥️ Streamlit App
Interactive app to predict ride fare using the trained model.

Run locally:
```bash
export PYTHONPATH=.
streamlit run app/app.py
