# Streamlit Employee Attrition Dashboard

This repository contains a Streamlit app `app.py` that:
- Shows 5 interactive HR charts with filters (JobRole multiselect and a satisfaction slider)
- Trains Decision Tree, Random Forest, and Gradient Boosting models and shows metrics (accuracy, precision, recall, F1, ROC-AUC, CV)
- Allows uploading a new dataset to predict Attrition and download predictions

How to deploy:
1. Create a GitHub repo, upload these files to the root.
2. Connect repo to Streamlit Cloud and set main file to `app.py`.

Dependencies: streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, seaborn
