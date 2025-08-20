📘 Student Score Prediction
📌 Project Overview

This project predicts a student’s final exam score based on their study hours and attendance percentage using a Machine Learning model.
It demonstrates how Linear Regression can be applied to real-world educational data to assist students and teachers.

❓ Problem Statement

Can we predict a student’s exam score if we know:

How many hours they studied 📚

Their attendance percentage 🎯

🔑 Features

✅ Import & clean CSV data
✅ Visualize data (study hours vs score, attendance vs score)
✅ Train & test split for evaluation
✅ Build regression model using scikit-learn
✅ Predict new student’s score with user input
✅ Evaluate model using R² Score & Mean Absolute Error
✅ Deployed on Streamlit for interactive use

🗂 Dataset (Sample)
Hours_Studied	Attendance	Final_Score
5	90	85
3	60	55
6	95	90
⚙ Tech Stack

Python 3

pandas, numpy → Data handling

matplotlib, seaborn → Data visualization

scikit-learn → Machine learning model

Streamlit → Web app deployment

🚀 How to Run
1️⃣ Clone Repository
git clone https://github.com/ZainabNehal/Student-Score-Prediction-Project
cd student-score-prediction

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py

📊 Expected Output

Enter Hours_Studied = 4 and Attendance = 80%


Deployed streamlit app link- https://student-score-prediction-project-ev6ucjuqgjxzoef42m7ruj.streamlit.app/
