# ✈️ SFO 2017 Passenger Experience Dashboard  

This project is an **interactive Streamlit dashboard** built on the **San Francisco International Airport (SFO) 2017 Customer Survey** dataset.  
The goal is to transform survey responses into **actionable insights** that highlight passenger satisfaction, pain points, and opportunities for operational and infrastructure improvements.  

👉 [View the app on Streamlit Community](#)  

---

## 📊 About the Dataset  
The survey was conducted in **May 2017** with thousands of passengers providing feedback on:  
- **Facilities & Services** (artwork, food, stores, signage, WiFi, parking, etc.)  
- **Cleanliness & Safety** (restrooms, boarding areas, overall airport cleanliness)  
- **Passenger Experience** (wayfinding, security screening, TSA PreCheck, check-in)  
- **Trip Context** (trip purpose, airline, boarding area, travel frequency, demographics)  

**Source:** [DataSF – SFO Customer Survey 2017](https://data.sfgov.org/Transportation/2017-SFO-Customer-Survey/nnh5-5rwz/about_data)  

---

## 🚀 Dashboard Features  
The dashboard is organized into **multiple tabs** for deeper exploration:  

- **Overview** – High-level KPIs, satisfaction scores, and passenger demographics.  
- **Passenger Flow & Operations** – Visuals on boarding areas, wait times, and TSA staffing impact.  
- **Cleanliness & Facilities** – Insights into restroom ratings, WiFi quality, and amenities.  
- **Food & Retail** – Feedback on affordability, variety, and demand for more vendors.  
- **Comments Explorer** – Sentiment and keyword analysis of passenger comments.  

**Tech Stack:**  
- [Streamlit](https://streamlit.io/)  
- [Plotly Express & Graph Objects](https://plotly.com/python/)  
- [Pandas & NumPy](https://pandas.pydata.org/)  
- [SciPy](https://scipy.org/) for hypothesis testing  

---

## ⚙️ Run Locally  

Clone this repo and install dependencies:  

```bash
git clone https://github.com/your-username/sfo-survey-dashboard.git
cd sfo-survey-dashboard
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 🧑‍💻 About This Case Study

This project demonstrates how data can guide airport experience improvements by combining data cleaning, statistical analysis, and interactive visualization.
It highlights the power of turning raw survey responses into insights that drive decisions.