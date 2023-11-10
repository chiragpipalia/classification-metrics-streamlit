import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
from sklearn import metrics

st.set_page_config(layout="wide")
st.title('Classification Metrics')
st.subheader('KS - ROC - PR')

size = 5000
scale = 5
def create_dataset(separation = 'med', size = 5000, scale = 5):
	# Generate two classes out of normal distributions - good classifier
	class1_x = np.random.normal(loc = 10, scale = scale, size = size)
	class1_y = np.random.normal(loc = 1, scale = scale, size = size)

	class2_x = np.random.normal(loc = 1, scale = scale, size = size)
	class2_y = np.random.normal(loc = 5, scale = scale, size = size)

	df_class1 = pd.DataFrame({'x' : class1_x, 'y' : class1_y, 'class': 0})
	df_class2 = pd.DataFrame({'x' : class2_x, 'y' : class2_y, 'class': 1})
	df = pd.concat([df_class1, df_class2])
	return df

## Get inputs
size_filter = st.slider('Size', min_value=1000, max_value=5000, step=500)
scale_filter = st.slider('Scale', min_value=1, max_value=15, step=5)


chart_data = create_dataset(size = size_filter,
	scale = scale_filter
	)

def get_train_test_split(df):
  X = df[['x', 'y']].values
  y = df[['class']].values
  train_samples = 1000  # Samples used for training the models
  X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      shuffle=True
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
  # Unite both visions to be able to filter
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
  return results


chart_data = create_dataset(scale = 5)
fig_data_good = px.scatter(
    chart_data,
    x="x",
    y="y",
    color="class"
)
y_proba_good, y_test_good = train_model_return_probs(chart_data)
df_res_good = pd.DataFrame({'y_test' : y_test_good.flatten(), 'y_proba': y_proba_good[:, 1]})
fig_prob_good =  px.histogram(df_res_good, x = 'y_proba',  color = 'y_test', nbins=25, barmode = 'group')
cdf_good = get_classes_cdf(y_test_good.flatten(), y_proba_good, ret_type = 'melt')
fig_cdf_good = px.line(cdf_good, x='proba' , y='cdf' , color='class')

chart_data_bad = create_dataset(scale = 15)
fig_data_bad = px.scatter(
    chart_data_bad,
    x="x",
    y="y",
    color="class"
)
y_proba_bad, y_test_bad = train_model_return_probs(chart_data_bad)
df_res_bad = pd.DataFrame({'y_test' : y_test_bad.flatten(), 'y_proba': y_proba_bad[:, 1]})
cdf_bad = get_classes_cdf(y_test_bad.flatten(), y_proba_bad, ret_type = 'melt')
fig_prob_bad =  px.histogram(df_res_bad, x = 'y_proba',  color = 'y_test', nbins=25, barmode = 'group')
fig_cdf_bad = px.line(cdf_bad, x='proba' , y='cdf' , color='class')


## AUC Curve
def get_fpr_tpr(y_test, y_proba):
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
  # fpr_tpr_vals = np.append(fpr, tpr)
  # fpr_tpr_labels = len(fpr)*['fpr'] + len(tpr)*[tpr]
  # pd.DataFrame({'fpr_tpr': fpr_tpr, 'fpr_tpr_labels' : fpr_tpr_labels})
  return fpr, tpr

fpr_good, tpr_good = get_fpr_tpr(y_test_good.ravel(), y_proba_good[:,1])
fpr_bad, tpr_bad = get_fpr_tpr(y_test_bad.ravel(), y_proba_bad[:,1])

fig_fpr_tpr_good = px.line(x=fpr_good , y=tpr_good)


col1, col2 = st.columns(2)

with col1:
	st.header("Good")
	st.plotly_chart(fig_data_good, theme="streamlit", use_container_width=True)
	st.subheader("Probability Dist")
	st.plotly_chart(fig_prob_good, theme="streamlit", use_container_width=True)
	st.plotly_chart(fig_cdf_good, theme="streamlit", use_container_width=True)
	st.plotly_chart(fig_fpr_tpr_good, theme="streamlit", use_container_width=True)
	
with col2:
	st.header("Bad")
	st.plotly_chart(fig_data_bad, theme="streamlit", use_container_width=True)
	# st.subheader("Probability Dist")
	st.plotly_chart(fig_prob_bad, theme="streamlit", use_container_width=True)
	st.plotly_chart(fig_cdf_bad, theme="streamlit", use_container_width=True)
