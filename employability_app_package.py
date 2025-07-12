# Folder: student-employability-prediction-app/

# File: student_employability_app_final.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import base64
import matplotlib.pyplot as plt

# --- Load Model & Scaler ---
model = joblib.load("employability_predictor.pkl")
scaler = joblib.load("scaler.pkl")

# --- Utility Functions ---
def generate_pdf_report(data, result, confidence):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Employability Prediction Report", ln=True, align="C")
    pdf.ln(10)

    for k, v in data.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)

    file_path = "prediction_report.pdf"
    pdf.output(file_path)
    return file_path


def get_pdf_download_link(file_path):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📄 Download PDF Report</a>'
    return href

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="Graduate Employability Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/UNESCO_logo.svg/2560px-UNESCO_logo.svg.png",
    width=200
)

st.sidebar.title("About This App")
st.sidebar.markdown("""
This app predicts **graduate employability** based on:
- GPA
- Internship
- Mock Interview
- Soft Skills
- Extracurricular

It uses a trained **SVM model** optimised with SMOTE.
Outputs: Prediction, confidence, feature insights, PDF report.

Developed for MSc Capstone Project.
""")

st.sidebar.info("Version: 2.0 | Last Updated: 2025-07-06")

# --- Header ---
st.title("🎓 Graduate Employability Dashboard")
st.subheader("Empowering HEIs with actionable, data-driven insights.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 Input Form", "📊 Feature Insights", "📄 Report"])

with tab1:
    st.header("📋 Student Profile Input")

    with st.form("input_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gpa = st.number_input("GPA (0–4.0)", 0.0, 4.0, value=3.0, step=0.01)
            soft_skills = st.slider("Soft Skills (0–100)", 0, 100, 75)

        with col2:
            internship = st.slider("Internship (0–100)", 0, 100, 80)
            extracurricular = st.slider("Extracurricular (0–100)", 0, 100, 60)

        with col3:
            mock_interview = st.slider("Mock Interview (0–100)", 0, 100, 70)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_data = np.array([[gpa, internship, mock_interview, soft_skills, extracurricular]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        confidence = abs(model.decision_function(input_scaled)[0])

        result = "✅ Employable" if prediction == 1 else "⚠️ At Risk"
        st.session_state['data'] = {
            "GPA": gpa,
            "Internship": internship,
            "Mock Interview": mock_interview,
            "Soft Skills": soft_skills,
            "Extracurricular": extracurricular
        }
        st.session_state['result'] = result
        st.session_state['confidence'] = confidence

        st.markdown("---")
        st.metric(label="Prediction", value=f"{result}", delta=f"Confidence: {confidence:.2f}")

with tab2:
    st.header("📊 Feature Contribution")

    if 'data' in st.session_state:
        df = pd.DataFrame([st.session_state['data']])
        df.T.plot(kind="barh", legend=False, figsize=(6, 3), color='skyblue')
        plt.xlabel("Feature Value")
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

with tab3:
    st.header("📄 Downloadable Prediction Report")

    if 'result' in st.session_state:
        pdf_path = generate_pdf_report(
            st.session_state['data'],
            st.session_state['result'],
            st.session_state['confidence']
        )
        st.markdown(get_pdf_download_link(pdf_path), unsafe_allow_html=True)
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

st.markdown("---")
st.caption("© 2025 Choong Muh In / APU University | Graduate Employability Prediction App | For research purposes only.")


# File: requirements.txt

"""
streamlit>=1.33
joblib>=1.4
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
fpdf>=1.7
scikit-learn>=1.4
"""

# File: .gitignore

"""
__pycache__/
*.pyc
*.pkl
*.joblib
*.pdf
.ipynb_checkpoints/
.env
*.env
.vscode/
.idea/
"""

# File: README.md

"""
# 🎓 Graduate Employability Prediction Dashboard

This app predicts **graduate employability** based on academic and experiential attributes, using a trained Support Vector Machine (SVM) model.

## 🚀 Features

✅ Predicts employability based on:
- GPA
- Internship
- Mock Interview
- Soft Skills
- Extracurricular Activities

✅ Outputs:
- Prediction (Employable / At Risk)
- Confidence score
- Feature contribution chart
- Downloadable PDF report

✅ Built with:
- Streamlit
- scikit-learn
- joblib
- pandas, numpy, matplotlib
- fpdf

---

## 🗂️ Project Structure

```
student-employability-prediction-app/
├── student_employability_app_final.py
├── employability_predictor.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Run

### Local Setup
1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Run the app:
```bash
streamlit run student_employability_app_final.py
```

---

### Deployment on [Streamlit Cloud](https://streamlit.io/cloud)
✅ Push the repository (with all files) to GitHub  
✅ Log in to Streamlit Cloud and connect the repo  
✅ Deploy — and you’re live!

---

## 📄 Notes
- Make sure both `employability_predictor.pkl` and `scaler.pkl` are present in the same folder as the Python script.
- PDF reports are saved in the app’s working directory.

---

## 📝 License
For educational/research purposes only.  
© 2025 Choong Muh In / APU University
"""
