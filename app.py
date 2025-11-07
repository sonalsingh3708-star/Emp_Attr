
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(layout="wide", page_title="Employee Attrition Dashboard")

st.title("HR Attrition Dashboard — Streamlit App")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("EA.csv")
        except Exception:
            df = pd.read_csv("EA_sample.csv")
    return df

# Sidebar
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset (same schema as EA.csv) or use sample", type=["csv"])
df = load_data(uploaded_file)
st.sidebar.markdown(f"**Rows:** {df.shape[0]}  \n**Columns:** {df.shape[1]}")

def prepare_target(y):
    if y.dtype == object or y.dtype.name == 'category':
        y = y.str.strip().str.lower().map({"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0}).fillna(y)
    if y.dtype == object:
        y = pd.factorize(y)[0]
    return y.astype(int)

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    for c in numeric_cols[:]:
        if X[c].nunique() <= 6:
            numeric_cols.remove(c)
            categorical_cols.append(c)
    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer([('num', num_transformer, numeric_cols), ('cat', cat_transformer, categorical_cols)], remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

tab1, tab2, tab3 = st.tabs(["Dashboard", "Model Training", "Predict & Download"])

with tab1:
    st.header("Interactive Dashboard (5 charts + filters)")
    job_roles = sorted(df['JobRole'].unique()) if 'JobRole' in df.columns else []
    selected_roles = st.multiselect("Filter by Job Role (multi-select)", options=job_roles, default=job_roles)
    if 'JobSatisfaction' in df.columns:
        sat_col = 'JobSatisfaction'
    else:
        sat_candidates = [c for c in df.columns if 'satisfaction' in c.lower()]
        sat_col = sat_candidates[0] if sat_candidates else None
    if sat_col:
        min_sat = int(df[sat_col].min()); max_sat = int(df[sat_col].max())
        sat_slider = st.slider(f"Filter by {sat_col}", min_value=min_sat, max_value=max_sat, value=(min_sat, max_sat))
    else:
        sat_slider = None
    dff = df.copy()
    if selected_roles:
        dff = dff[dff['JobRole'].isin(selected_roles)]
    if sat_col and sat_slider is not None:
        dff = dff[(dff[sat_col] >= sat_slider[0]) & (dff[sat_col] <= sat_slider[1])]

    if 'Attrition' in dff.columns and 'JobRole' in dff.columns:
        gr = dff.groupby('JobRole')['Attrition'].apply(lambda x: (x.astype(str).str.lower()=='yes').mean() if x.dtype==object else (x==1).mean()).reset_index(name='attrition_rate')
        fig1 = px.bar(gr, x='JobRole', y='attrition_rate', title='Attrition rate by Job Role', labels={'attrition_rate':'Attrition Rate'}, hover_data={'attrition_rate':':.2f'})
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Attrition or JobRole column missing - cannot show chart 1.")

    if sat_col and 'MonthlyIncome' in dff.columns:
        dff['Attrition_flag'] = dff['Attrition'].apply(lambda x: 1 if str(x).strip().lower()=='yes' or x==1 else 0)
        fig2 = px.scatter(dff, x=sat_col, y='MonthlyIncome', color='Attrition_flag', hover_data=['JobRole'], title=f'{sat_col} vs MonthlyIncome (colored by Attrition)')
        st.plotly_chart(fig2, use_container_width=True)

    if 'YearsAtCompany' in dff.columns and 'Attrition' in dff.columns:
        fig3 = px.histogram(dff, x='YearsAtCompany', color='Attrition', barmode='group', title='Years at Company distribution by Attrition')
        st.plotly_chart(fig3, use_container_width=True)

    numeric = dff.select_dtypes(include=['int64','float64']).corr()
    if not numeric.empty:
        corr_small = numeric.loc[numeric.columns[:12], numeric.columns[:12]]
        fig4 = px.imshow(corr_small, text_auto=True, title='Correlation matrix (top numeric cols)')
        st.plotly_chart(fig4, use_container_width=True)

    if 'OverTime' in dff.columns and 'JobRole' in dff.columns and 'Attrition' in dff.columns:
        ct = pd.crosstab([dff['JobRole'], dff['OverTime']], dff['Attrition'])
        ct = ct.reset_index().melt(id_vars=['JobRole','OverTime'], var_name='Attrition', value_name='count')
        fig5 = px.bar(ct, x='JobRole', y='count', color='Attrition', facet_col='OverTime', title='Attrition counts by JobRole and OverTime (facet)')
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("**Actionable insights (auto-generated):**")
    with st.expander("View suggested actions"):
        st.write("- Focus on job roles with high attrition rates from Chart 1. Consider targeted retention programs.\n- Investigate employees with low job satisfaction & low income (Chart 2).\n- If OverTime correlates with attrition for specific roles (Chart 5), examine workload & staffing.\n- Use YearsAtCompany patterns (Chart 3) to enhance onboarding & first 3-year retention efforts.")

with tab2:
    st.header("Model Training & Evaluation (Decision Tree, Random Forest, GBRT)")
    st.write("This tab trains three models on the current dataset. Click **Train Models** to run.")

    train_button = st.button("Train Models (DecisionTree / RandomForest / GradientBoosting)")
    if train_button:
        if 'Attrition' not in df.columns:
            st.error("Attrition column not found in dataset.")
        else:
            data = df.copy()
            X = data.drop(columns=['Attrition']).copy()
            y = prepare_target(data['Attrition'])
            preprocessor, num_cols, cat_cols = build_preprocessor(X)
            models = {
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
            }
            pipelines = {name: Pipeline([('pre', preprocessor), ('clf', model)]) for name, model in models.items()}
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

            results = []
            importances = {}
            for name, pipe in pipelines.items():
                with st.spinner(f"Training {name}..."):
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    try:
                        y_proba = pipe.predict_proba(X_test)[:,1]
                    except:
                        y_proba = pipe.decision_function(X_test)
                    train_pred = pipe.predict(X_train)
                    try:
                        train_proba = pipe.predict_proba(X_train)[:,1]
                    except:
                        train_proba = pipe.decision_function(X_train)
                    metrics = {
                        'model': name,
                        'train_accuracy': accuracy_score(y_train, train_pred),
                        'test_accuracy': accuracy_score(y_test, y_pred),
                        'train_precision': precision_score(y_train, train_pred, zero_division=0),
                        'test_precision': precision_score(y_test, y_pred, zero_division=0),
                        'train_recall': recall_score(y_train, train_pred, zero_division=0),
                        'test_recall': recall_score(y_test, y_pred, zero_division=0),
                        'train_f1': f1_score(y_train, train_pred, zero_division=0),
                        'test_f1': f1_score(y_test, y_pred, zero_division=0),
                        'train_auc': roc_auc_score(y_train, train_proba),
                        'test_auc': roc_auc_score(y_test, y_proba)
                    }
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    metrics['cv_acc_mean'] = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy').mean()
                    try:
                        metrics['cv_auc_mean'] = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc').mean()
                    except:
                        metrics['cv_auc_mean'] = np.nan
                    results.append(metrics)
                    model_obj = pipe.named_steps['clf']
                    feat_names = []
                    try:
                        num_names = num_cols
                        cat_names = list(pipe.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
                        feat_names = list(num_names) + cat_names
                    except Exception:
                        feat_names = []
                    if hasattr(model_obj, 'feature_importances_') and feat_names:
                        importances[name] = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)

            res_df = pd.DataFrame(results).set_index('model').round(4)
            st.subheader("Model comparison table")
            st.dataframe(res_df)

            st.subheader("ROC curves on test set")
            fig = go.Figure()
            for name, pipe in pipelines.items():
                try:
                    y_proba = pipe.predict_proba(X_test)[:,1]
                except:
                    y_proba = pipe.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves (Test Set)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Confusion Matrices")
            for name, pipe in pipelines.items():
                y_pred = pipe.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=['Actual 0 (Stay)','Actual 1 (Leave)'], columns=['Pred 0 (Stay)','Pred 1 (Leave)'])
                st.write(f"**{name}**")
                st.dataframe(cm_df)

            st.subheader("Top feature importances (per model)")
            for name, imp in importances.items():
                st.write(f"**{name}**")
                st.bar_chart(imp)

            st.session_state['trained_pipelines'] = pipelines
            st.session_state['last_results'] = res_df
            st.success("Training complete. Models saved in session for predictions.")

with tab3:
    st.header("Upload new data and predict Attrition")
    upload_new = st.file_uploader("Upload new dataset for prediction", type=['csv'])
    model_choice = st.selectbox("Choose model for prediction (requires training first or will train automatically):", options=['RandomForest','GradientBoosting','DecisionTree'])
    predict_button = st.button("Predict & Download")

    if upload_new is not None and predict_button:
        new_df = pd.read_csv(upload_new)
        if 'trained_pipelines' not in st.session_state:
            st.warning("Models not trained in this session — training automatically on full dataset now.")
            if 'Attrition' not in df.columns:
                st.error("Cannot auto-train because original dataset doesn't have Attrition column.")
            else:
                data = df.copy()
                X = data.drop(columns=['Attrition']).copy()
                y = prepare_target(data['Attrition'])
                preprocessor, num_cols, cat_cols = build_preprocessor(X)
                models = {
                    'DecisionTree': DecisionTreeClassifier(random_state=42),
                    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
                    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
                }
                pipelines = {name: Pipeline([('pre', preprocessor), ('clf', model)]) for name, model in models.items()}
                for name, pipe in pipelines.items():
                    pipe.fit(X, y)
                st.session_state['trained_pipelines'] = pipelines

        pipelines = st.session_state['trained_pipelines']
        chosen_pipe = pipelines[model_choice]
        preds = chosen_pipe.predict(new_df)
        pred_labels = pd.Series(preds).map({1:'Yes',0:'No'})
        out_df = new_df.copy()
        out_df['Predicted_Attrition'] = pred_labels.values
        csv = out_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("Prediction completed. Download using the link above.")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload EA.csv as 'EA.csv' or use the sample file.")

