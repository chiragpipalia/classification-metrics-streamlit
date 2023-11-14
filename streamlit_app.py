import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
from scipy import stats

st.set_page_config(layout="wide")
st.title('Classification Metrics')
st.subheader('KS - ROC - PR')
st.text("Effects of class imbalance and separation on various classification metrics.")
st.text("We will create 2 datasets, Baseline and Model. Adjust class separation and data imbalance to compare its effects on the metrics.")

def create_dataset(size = 10000, separation = 1, scale = 2, imbalance = False, minority_ratio = 0.5):
	# separation = 1
	center = 5
	class0_base, class1_base = center-(center*separation) , center+(center*separation)

	if imbalance:
		# Generate two classes out of normal distributions - good classifier
		class0_x = np.random.normal(loc = class0_base, scale = scale, size = int(size*(1-minority_ratio)))
		class0_y = np.random.normal(loc = class0_base, scale = scale, size = int(size*(1-minority_ratio)))

		class1_x = np.random.normal(loc = class1_base, scale = scale, size = int(size*minority_ratio))
		class1_y = np.random.normal(loc = class1_base, scale = scale, size = int(size*minority_ratio))
	else:
		size = int(size*minority_ratio)
		class0_x = np.random.normal(loc = class0_base, scale = scale, size = size)
		class0_y = np.random.normal(loc = class0_base, scale = scale, size = size)

		class1_x = np.random.normal(loc = class1_base, scale = scale, size = size)
		class1_y = np.random.normal(loc = class1_base, scale = scale, size = size)


	df_class0 = pd.DataFrame({'x' : class0_x, 'y' : class0_y, 'class': 0})
	df_class1 = pd.DataFrame({'x' : class1_x, 'y' : class1_y, 'class': 1})
	df = pd.concat([df_class0, df_class1])
	df.sort_values(by='class', inplace = True)
	return df

def get_train_test_split(df):
  X = df[['x', 'y']].values
  y = df[['class']].values
  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size = 0.33,
      stratify=y
  )
  return X_train, X_test, y_train, y_test

def train_model_return_probs(df):
  X_train, X_test, y_train, y_test = get_train_test_split(df)
  model_good = GaussianNB()
  model_good.fit(X_train, y_train)
  y_pred_good = model_good.predict(X_test)
  y_proba_good = model_good.predict_proba(X_test)

  return y_proba_good, y_test

def cdf(sample, x, sort = False):
    '''
    Return the value of the Cumulative Distribution Function, evaluated for a given sample and a value x.

    Args:
        sample: The list or array of observations.
        x: The value for which the numerical cdf is evaluated.

    Returns:
        cdf = CDF_{sample}(x)
    '''

    # Sorts the sample, if needed
    if sort:
        sample.sort()

    # Counts how many observations are below x
    cdf = sum(sample <= x)

    # Divides by the total number of observations
    cdf = cdf / len(sample)

    return cdf

def get_classes_cdf(y_real, y_proba, ret_type = 'melt'):
  '''
  Return cdf, proba for each class and KS statistic

  Args:
  	y_real: List or array of y truths
  	y_proba: List or array of predicted probabilities by the model
  	ret_type: melt, returns cdf and probabilities for both classes in a single column with column class for mapping; else 
  			  returns cdf and probabilities for each class in a separate column
  Returns:
  		results: dict of cdfs and probabilities
  		ks: KS statistics
  ''' 
  df = pd.DataFrame()
  df['real'] = y_real
  df['proba'] = y_proba[:, 1]

  # Recover each class
  class0 = df[df['real'] == 0].sort_values('proba', ascending = False)
  class1 = df[df['real'] == 1].sort_values('proba', ascending = False)

  # Calculates the cdfs
  cdf0 = np.array([cdf(class0['proba'].values, x, sort = False) for x in class0['proba'].values])
  cdf1 = np.array([cdf(class1['proba'].values, x, sort = False) for x in class1['proba'].values])
  color = len(cdf0)*[0] + len(cdf1)*[1]
  ks = round(stats.ks_2samp(class0['proba'].values, class1['proba'].values).statistic, 4)

  if ret_type == 'melt':
  	results = {
		'cdf': np.append(cdf0, cdf1),
		'proba': np.append(class0['proba'].values, class1['proba'].values),
		'class': color}
  else:
    results = {
        'cdf0': cdf0,
        'cdf1': cdf1,
        'proba0': class0['proba'].values,
        'proba1': class1['proba'].values}
  return results, ks


baseline_col, model_col = st.columns(2)
with baseline_col:
	st.header("Baseline")
	minority_ratio_baseline = 0.5
	imbalance_baseline = st.toggle('Make Base Imbalance')
	if imbalance_baseline:
		minority_ratio_baseline = st.slider('Minority ratio (baseline)', min_value=0.01, max_value=0.5, step=0.01, value=0.5)

	scale =  2
	separation_baseline = st.slider('Separation (baseline)', min_value=0.0, max_value=1.0, step=0.1, value = 1.0)

with model_col:
	st.header("Model")
	minority_ratio_model = 0.5
	imbalance_model = st.toggle('Make Modeled Imbalance')
	if imbalance_model:
		minority_ratio_model = st.slider('Minority ratio (model)', min_value=0.01, max_value=0.5, step=0.01, value=0.5)

	scale =  2
	separation_model = st.slider('Separation (model)', min_value=0.0, max_value=1.0, step=0.1, value = 1.0)

chart_data = create_dataset(separation = separation_baseline, scale = scale, imbalance = imbalance_baseline, minority_ratio = minority_ratio_baseline)
fig_data_good = px.scatter(
    chart_data,
    x="x",
    y="y",
    color="class"
)
y_proba_good, y_test_good = train_model_return_probs(chart_data)
df_res_good = pd.DataFrame({'class' : y_test_good.flatten(), 'y_proba': y_proba_good[:, 1]})
df_res_good.sort_values(by='class', inplace=True)
fig_prob_good =  px.histogram(df_res_good, x = 'y_proba',  color = 'class', nbins=25, barmode = 'group')
cdf_good, ks_good = get_classes_cdf(y_test_good.flatten(), y_proba_good, ret_type = 'melt')
fig_cdf_good = px.line(cdf_good, x='proba' , y='cdf' , color='class')

chart_data_bad = create_dataset(separation = separation_model, scale = scale, imbalance = imbalance_model, minority_ratio = minority_ratio_model)
fig_data_bad = px.scatter(
    chart_data_bad,
    x="x",
    y="y",
    color="class"
)

y_proba_bad, y_test_bad = train_model_return_probs(chart_data_bad)
df_res_bad = pd.DataFrame({'class' : y_test_bad.flatten(), 'y_proba': y_proba_bad[:, 1]})
df_res_bad.sort_values(by='class', inplace=True)
fig_prob_bad =  px.histogram(df_res_bad, x = 'y_proba',  color = 'class', nbins=25, barmode = 'group')
cdf_bad, ks_bad = get_classes_cdf(y_test_bad.flatten(), y_proba_bad, ret_type = 'melt')
fig_cdf_bad = px.line(cdf_bad, x='proba' , y='cdf' , color='class')


## AUC Curve
def get_fpr_tpr(y_test, y_proba):
  fpr, tpr, thresholds = roc_curve(y_test, y_proba)
  # fpr_tpr_vals = np.append(fpr, tpr)
  # fpr_tpr_labels = len(fpr)*['fpr'] + len(tpr)*[tpr]
  # pd.DataFrame({'fpr_tpr': fpr_tpr, 'fpr_tpr_labels' : fpr_tpr_labels})
  return fpr, tpr

fpr_good, tpr_good = get_fpr_tpr(y_test_good.ravel(), y_proba_good[:,1])
roc_auc_score_good = round(auc(fpr_good, tpr_good), 2)
df_fpr_tpr_good = pd.DataFrame({'fpr' : fpr_good, 'tpr' : tpr_good})
fpr_bad, tpr_bad = get_fpr_tpr(y_test_bad.ravel(), y_proba_bad[:,1])
roc_auc_score_bad = round(auc(fpr_bad, tpr_bad), 2)
df_fpr_tpr_bad = pd.DataFrame({'fpr' : fpr_bad, 'tpr' : tpr_bad})

x = y = [0, 0.5 ,1]
fig_fpr_tpr_good = px.line(df_fpr_tpr_good, x='fpr' , y='tpr')
fig_fpr_tpr_good.add_scatter(x=x, y=y, mode='lines', name='random', line=dict(color="#ffe476"))

fig_fpr_tpr_bad = px.line(df_fpr_tpr_bad, x='fpr' , y='tpr')
fig_fpr_tpr_bad.add_scatter(x=x, y=y, mode='lines', name='random', line=dict(color="#ffe476"))

# calculate the precision-recall auc
precision_good, recall_good, _ = precision_recall_curve(y_test_good.ravel(), y_proba_good[:,1])
pr_auc_score_good = round(auc(recall_good, precision_good), 2)
df_pr_good = pd.DataFrame({'recall' : recall_good, 'precision' : precision_good})
fig_pr_good = px.line(df_pr_good, x='recall' , y='precision')

precision_bad, recall_bad, _ = precision_recall_curve(y_test_bad.ravel(), y_proba_bad[:,1])
pr_auc_score_bad =  round(auc(recall_bad, precision_bad), 2)
df_pr_bad = pd.DataFrame({'recall' : recall_bad, 'precision' : precision_bad})
fig_pr_bad = px.line(df_pr_bad, x='recall' , y='precision')

st.subheader("Simulated data")
col1, col2 = st.columns(2)
with col1:
	st.plotly_chart(fig_data_good, theme="streamlit", use_container_width=True)
with col2:
	st.plotly_chart(fig_data_bad, theme="streamlit", use_container_width=True)

st.subheader("Predicted Probability Dist")
col1_predprob, col2_predprob = st.columns(2)
with col1_predprob:
	st.plotly_chart(fig_prob_good, theme="streamlit", use_container_width=True)
with col2_predprob:
	st.plotly_chart(fig_prob_bad, theme="streamlit", use_container_width=True)


## KS
st.subheader("Kolmogorov-Smirnov (KS)")
st.write("""It's a measure of the discriminatory power of a binary classification model. 
	It's often used in credit risk modeling. 
	The KS statistic represents the ***maximum vertical distance*** between the cumulative distributions of the positive and negative classes in the predicted probabilities. 
	A higher KS value indicates better model discrimination.""")
col1_ks, col2_ks = st.columns(2)
with col1_ks:
	st.metric("K-S Statistic", ks_good)
	st.plotly_chart(fig_cdf_good, theme="streamlit", use_container_width=True)
with col2_ks:
	# st.subheader("KS")
	st.metric("K-S Statistic", ks_bad)
	st.plotly_chart(fig_cdf_bad, theme="streamlit", use_container_width=True)

## ROC
st.subheader("ROC - AUC")
st.write("""This is a performance metric for binary classification problems at various threshold settings. 
	The ROC curve is a graphical representation of the true positive rate against the false positive rate. 
	AUC-ROC measures the area under this curve. A higher AUC-ROC value (closer to 1) suggests better discrimination and model performance.""")
st.markdown("**Use Cases:**")
st.write("""Commonly used in scenarios where the ***cost of false positives and false negatives is roughly equal***, 
	and the overall classification performance is of interest.""")
col1_roc, col2_roc = st.columns(2)
with col1_roc:	
	st.metric("ROC AUC", roc_auc_score_good, round(roc_auc_score_good-0.5, 2))
	st.plotly_chart(fig_fpr_tpr_good, theme="streamlit", use_container_width=True)
with col2_roc:	
	st.metric("ROC AUC", roc_auc_score_bad, round(roc_auc_score_bad-0.5, 2))
	st.plotly_chart(fig_fpr_tpr_bad, theme="streamlit", use_container_width=True)

## PR	
st.subheader("Precision Recall (PR)")
st.write("""These are metrics used for evaluating the performance of binary classification models, especially when dealing with imbalanced datasets. 
	Precision is the ratio of true positive predictions to the total predicted positives, while recall (or sensitivity) is the ratio of true positives to the total actual positives. 
	Precision-Recall curves provide insight into a model's ability to correctly identify positive instances and minimize false positives.""")
st.markdown("**Use Cases:**")
st.write("""Particularly useful when the ***class distribution is imbalanced***, and the focus is on the positive class, 
	such as in fraud detection or rare disease diagnosis.""")
col1_pr, col2_pr = st.columns(2)
with col1_pr:
	st.metric("PR AUC", pr_auc_score_good, round(pr_auc_score_good-0.5, 2))
	st.plotly_chart(fig_pr_good, theme="streamlit", use_container_width=True)
with col2_pr:
	st.metric("PR AUC", pr_auc_score_bad, round(pr_auc_score_bad-0.5, 2))
	st.plotly_chart(fig_pr_bad, theme="streamlit", use_container_width=True)
