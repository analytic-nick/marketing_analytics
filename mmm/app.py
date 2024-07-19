#!/usr/bin/env python
# coding: utf-8

# # STEP 0.0 Set Up Environment

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib import pyplot as plt
import dash_bootstrap_components as dbc
import dash_daq as daq
from plotnine import *
import plotly.express as px

import pandas as pd
import numpy as np

import pathlib
from joblib import load

import json


# Load Data

# In[3]:


#simulated_data_df = pd.read_csv("C:/Users/nfole/OneDrive/Desktop/lab_62_mmm_python/data/de_simulated_data.csv")
simulated_data_df  = pd.read_csv("https://raw.githubusercontent.com/analytic-nick/marketing_analytics/main/mmm/data/de_simulated_data.csv")


# # STEP 1.0 DATA UNDERSTANDING 

#  Wrangle Data 
# 

# In[6]:


simulated_data_df.columns = simulated_data_df.columns.str.lower()

simulated_data_df['date'] = pd.to_datetime(simulated_data_df['date'])


#  Visualize Time Series Data

# In[9]:


simulated_data_df \
    .melt(
        id_vars="date"
    ) \
    .pipe(
        px.line,
        x              = "date", 
        y              = "value", 
        color          = "variable", 
        facet_col      = "variable", 
        facet_col_wrap = 2,
        template       = "plotly_dark"
    ) \
    .update_yaxes(matches=None)


# Visualize Current Spend Profile

# In[13]:


media_spend_df = simulated_data_df \
    .filter(regex = '_s$', axis = 1)\
    .sum() \
    .to_frame() \
    .reset_index() \
    .set_axis(['media', 'spend_current'], axis = 1)

media_spend_df \
    .pipe(
        px.bar,
        x = 'media', 
        y = 'spend_current',
        color = 'media',
        template = "plotly_dark"
    )


# Add/Remove Features for Modeling

# In[15]:


df = simulated_data_df \
    .assign(date_month = lambda x: x['date'].dt.month_name()) \
    .drop([ "search_clicks_p", "facebook_i"], axis = 1)



# # STEP 2.0 MODELING WITH ADSTOCK -

# Create Adstock Function

# In[21]:


def adstock(series, rate):

    tt = np.empty(len(series))
    tt[0] = series[0]

    for i in range(1, len(series)):
      tt[i] = series[i] + tt[i-1] * rate

    tt_series = pd.Series(tt, index=series.index)
    tt_series.name = "adstock_" + str(series.name)

    return tt_series


# Calculate Adstock

# In[24]:


adstock_tv_s = adstock(df.tv_s, 0.5)
adstock_ooh_s = adstock(df.ooh_s, 0.5)
adstock_print_s = adstock(df.print_s, 0.5)
adstock_search_s = adstock(df.search_s, 0.5)
adstock_facebook_s = adstock(df.facebook_s, 0.5)


# Sklearn 

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Split Test and Train

# In[30]:


X = pd.concat([adstock_tv_s, adstock_ooh_s, adstock_print_s, adstock_search_s, adstock_facebook_s,df.date, df.competitor_sales_b, pd.get_dummies(df.date_month)], axis=1) 
X.set_index('date', inplace=True) 

y = df.revenue

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_test.shape



# In[32]:


df2=X \
.filter(regex = '_s$' , axis = 1)
df2 = df2.reset_index()

df2=df2\
    .melt(
        id_vars="date"
    ) 


# Build Model

# In[35]:


pipeline_lm = Pipeline(
    [
        ('lm', LinearRegression())
    ]
)

pipeline_lm.fit(X_train, y_train)

pipeline_lm.score(X_test, y_test)

y_pred_test = pipeline_lm.predict(X_test)
y_pred_train = pipeline_lm.predict(X_train)


# Score Predictions on Test Set

# In[38]:


mean_absolute_error(y_test, y_pred_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_score(y_test, y_pred_test)


# In[40]:


# Coefficients to the Model

pipeline_lm['lm'].intercept_
pipeline_lm['lm'].coef_


# Search Adstock Rate (Optimize for Model Fit)

# In[43]:


rng = np.random.default_rng(123)
size = 100
max_adstock = 1
adstock_grid_df = pd.DataFrame(dict(
    adstock_tv_s = rng.uniform(0, max_adstock, size = size), 
    adstock_ooh_s = rng.uniform(0, max_adstock, size = size),
    adstock_print_s = rng.uniform(0, max_adstock, size = size),
    adstock_search_s = rng.uniform(0, max_adstock, size = size),
    adstock_facebook_s = rng.uniform(0, max_adstock, size = size),

))
adstock_grid_df

def adstock_search(df, grid, verbose = False):

    best_model = dict(
        model = None,
        params = None,
        score = None,
    )

    for tv, ooh, prnt, search, facebook in zip(grid.adstock_tv_s, grid.adstock_ooh_s, grid.adstock_print_s, grid.adstock_search_s, grid.adstock_facebook_s):

        adstock_tv_s = adstock(df.tv_s, tv)
        adstock_ooh_s = adstock(df.ooh_s, ooh)
        adstock_print_s = adstock(df.print_s, prnt)
        adstock_search_s = adstock(df.search_s, search)
        adstock_facebook_s = adstock(df.facebook_s, facebook)

        X = pd.concat([adstock_tv_s, adstock_ooh_s, adstock_print_s, adstock_search_s, adstock_facebook_s, df.competitor_sales_b, pd.get_dummies(df.date_month)], axis=1) 

        y = df.revenue

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        pipeline_lm = Pipeline(
            [
                ('lm', LinearRegression())
            ]
        )

        pipeline_lm.fit(X_train, y_train)

        score = pipeline_lm.score(X_test, y_test)

        if best_model['model'] is None or score > best_model['score']:
            best_model['model'] = pipeline_lm,
            best_model['params'] = dict(
                tv = tv,
                ooh = ooh,
                prnt = prnt, 
                search = search,
                facebook = facebook
            )
            best_model['score'] = score

            if verbose:
                print("New Best Model:")
                print(best_model)
                print("\n")
    
    if verbose:
        print("Done!")

    return best_model


# # STEP 3.0: BUDGET REBALANCE 

# Get Model Coeffiecients

# In[47]:


best_model = adstock_search(df, adstock_grid_df, verbose= True)


# Media Spend Rebalanced
# 

# In[49]:


best_model_coef = best_model['model'][0]['lm'].coef_
best_model_coef_names = X.columns

rebalancing_coef_df = pd.DataFrame(dict(
        name = best_model_coef_names, 
        value = best_model_coef
    )) \
    .set_index('name') \
    .filter(regex = "_s$", axis=0) \
    .assign(value = lambda x: x['value'] / np.sum(x['value'])) \
    .reset_index()


total_current_spend = media_spend_df['spend_current'].sum()

media_spend_rebalanced_df = media_spend_df.copy()
media_spend_rebalanced_df['spend_new'] = total_current_spend * (rebalancing_coef_df['value'])

media_spend_rebalanced_df
  
media_spend_rebalanced_df \
    .melt(
        id_vars='media'
    ) \
    .pipe(
        px.bar,
        x = 'variable', 
        y = 'value',
        color = 'media', 
        barmode = 'group',
        template = 'plotly_dark'
    ) 
    


# In[52]:


df3=pd.merge(df2,rebalancing_coef_df, how='left',left_on=['variable'],right_on=['name'])
df3["contribution"]=df3.value_x * df3.value_y


# In[54]:


df4 = pd.DataFrame(df3, columns=['date', 'variable', 'contribution'])\
    .set_index('date')\
    .sort_values('date')

df4["label"]=df4.variable.str.replace('adstock_', '')

pt = pd.pivot_table(df4, columns=['label'], index=['date'], values=['contribution'], fill_value=0)

df4 = df4.reset_index()
fig_area=pt.plot.area()


# Predicted Revenue after Rebalancing
# 

# In[57]:


X_train_adstock = X_train[X_train.columns[X_train.columns.str.startswith('adstock_')]]

X_train_not_adstock = X_train[X_train.columns[~X_train.columns.str.startswith('adstock_')]]

X_train_adstock_rebal = X_train_adstock.copy()
X_train_adstock_rebal.sum(axis = 1)

for i, col in enumerate(X_train_adstock_rebal.columns):
    X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]

X_train_adstock_rebal

X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)

predicted_revenue_current = best_model['model'][0].predict(X_train).sum()

predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()

predicted_revenue_new / predicted_revenue_current


# # STEP 4.0 BUDGET OPTIMIZATION 
# 

# Search for optimal budget for media channels

# In[61]:


rng = np.random.default_rng(123)
size = 1000
budget_grid_df = pd.DataFrame(dict(
    adstock_tv_s = rng.uniform(0, 1, size = size), 
    adstock_ooh_s = rng.uniform(0, 1, size = size),
    adstock_print_s = rng.uniform(0, 1, size = size),
    adstock_search_s = rng.uniform(0, 1, size = size),
    adstock_facebook_s = rng.uniform(0, 1, size = size),
))


#for i, row in enumerate(budget_grid_df.index):
#    print(budget_grid_df.loc[i, :] / np.sum(budget_grid_df.loc[i, :]))


# Function to calculate new media spend numbers

# In[64]:


def optimize_budget(df, grid, media_spend, verbose = True):

    X_train = df
    budget_grid_df = grid
    media_spend_df = media_spend

    best_budget = dict(
        rebalancing_coef = None,
        media_spend_rebal = None,
        score = None,
    )

    for i, row in enumerate(budget_grid_df.index):
        
        # Scale the random budget mix
        budget_scaled = budget_grid_df.loc[i, :] / np.sum(budget_grid_df.loc[i, :])

        # print(budget_scaled)

        # Create rebalencing coefficients
        rebalancing_coef_df = budget_scaled \
            .to_frame() \
            .reset_index() \
            .set_axis(['name', 'value'], axis = 1)

        # Rebalance Adstock
        X_train_adstock = X_train[X_train.columns[X_train.columns.str.startswith('adstock_')]]

        X_train_not_adstock = X_train[X_train.columns[~X_train.columns.str.startswith('adstock_')]]

        X_train_adstock_rebal = X_train_adstock.copy()
        X_train_adstock_rebal.sum(axis = 1)

        for i, col in enumerate(X_train_adstock_rebal.columns):
            X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]

        X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)

        # Make Predictions
        predicted_revenue_current = best_model['model'][0].predict(X_train).sum()

        predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()
        
        score = predicted_revenue_new / predicted_revenue_current

        # Media Spend Rebalanced
        total_current_spend = media_spend_df['spend_current'].sum()
        media_spend_rebalanced_df = media_spend_df.copy()
        media_spend_rebalanced_df['spend_new'] = total_current_spend * (rebalancing_coef_df['value'])

        if best_budget['score'] is None or score > best_budget['score']:
            best_budget['rebalancing_coef'] = rebalancing_coef_df,
            best_budget['media_spend_rebal'] = media_spend_rebalanced_df
            best_budget['score'] = score

            if verbose:
                print("New Best Budget:")
                print(best_budget)
                print("\n")
    
    if verbose:
        print("Done!")

    return best_budget


# In[66]:


budget_optimized = optimize_budget(
    df          = X_train,
    grid        = budget_grid_df, 
    media_spend =  media_spend_df,
    verbose     = True
)

budget_optimized['media_spend_rebal'] \
    .melt(
        id_vars='media'
    ) \
    .pipe(
        px.bar,
        x = 'variable', 
        y = 'value',
        color = 'media', 
        barmode = 'group',
        template = 'plotly_dark'
    ) 


# # STEP 5.0 App

# In[81]:


# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)
server = app.server

PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
LOGO = "https://www.business-science.io/img/business-science-logo.png"

# ARTIFACTS


# Functions

# In[84]:


def generate_grid(size = 10):
    '''Generates a pandas DataFrame containin random budget media mix portfolios'''
    rng = np.random.default_rng(123)
    budget_grid_df = pd.DataFrame(dict(
        adstock_tv_s = rng.uniform(0, 1, size = size), 
        adstock_ooh_s = rng.uniform(0, 1, size = size),
        adstock_print_s = rng.uniform(0, 1, size = size),
        adstock_search_s = rng.uniform(0, 1, size = size),
        adstock_facebook_s = rng.uniform(0, 1, size = size),
    ))
    return budget_grid_df

def optimize_budget(df, grid, media_spend, verbose = True):
    '''Optimizes the budget'''
    
    X_train = df
    budget_grid_df = grid
    media_spend_df = media_spend

    best_budget = dict(
        rebalancing_coef = None,
        media_spend_rebal = None,
        score = None,
    )

    for i, row in enumerate(budget_grid_df.index):
        
        # Scale the random budget mix
        budget_scaled = budget_grid_df.loc[i, :] / np.sum(budget_grid_df.loc[i, :])

        # print(budget_scaled)

        # Create rebalencing coefficients
        rebalancing_coef_df = budget_scaled \
            .to_frame() \
            .reset_index() \
            .set_axis(['name', 'value'], axis = 1)

        # Rebalance Adstock
        X_train_adstock = X_train[X_train.columns[X_train.columns.str.startswith('adstock_')]]

        X_train_not_adstock = X_train[X_train.columns[~X_train.columns.str.startswith('adstock_')]]

        X_train_adstock_rebal = X_train_adstock.copy()
        X_train_adstock_rebal.sum(axis = 1)

        for i, col in enumerate(X_train_adstock_rebal.columns):
            X_train_adstock_rebal[col] = X_train_adstock_rebal.sum(axis = 1) * rebalancing_coef_df['value'][i]

        X_train_rebal = pd.concat([X_train_adstock_rebal, X_train_not_adstock], axis = 1)

        # Make Predictions
        predicted_revenue_current = best_model['model'][0].predict(X_train).sum()

        predicted_revenue_new = best_model['model'][0].predict(X_train_rebal).sum()
        
        score = predicted_revenue_new / predicted_revenue_current

        # Media Spend Rebalanced
        total_current_spend = media_spend_df['spend_current'].sum()
        media_spend_rebalanced_df = media_spend_df.copy()
        media_spend_rebalanced_df['spend_new'] = total_current_spend * rebalancing_coef_df['value']

        if best_budget['score'] is None or score > best_budget['score']:
            best_budget['rebalancing_coef'] = rebalancing_coef_df,
            best_budget['media_spend_rebal'] = media_spend_rebalanced_df
            best_budget['score'] = score

            if verbose:
                print("New Best Budget:")
                print(best_budget)
                print("\n")
    
    if verbose:
        print("Done!")

    return best_budget

fig2 = px.area(df4, x="date", y="contribution", color="label", template='plotly_dark',title="Contribution by Media Channel",height=400)


# Build the app

# In[87]:


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Marketing Mix Optimization", className="ml-2")),
                ],
                align="center"
               # no_gutters=True,
            ),
            
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            id="navbar-collapse", navbar=True, is_open=False
        ),
    ],
    color="dark",
    dark=True,
)


app.layout = html.Div(
    children = [
        navbar, 
        dbc.Row(
            [
                dbc.Col(
                    [
                        
                        html.H3("Welcome to the MMM Optimization Dashboard"),
                        html.Div(
                            id="intro",
                            children="Improve Estimated Revenue by optimizing media channel spend and budget.",
                        ),
                        html.Br(),
                        html.Hr(),
                        html.H5("Generate Random Media Spend Portfolios"),
                        html.P("Increase the number of media portfolios generated to improve the optimization results."),
                        dcc.Slider(
                            id = "budget-size", 
                            min = 1,
                            max = 5000,
                            step = 10,
                            marks = {
                                1: "1",
                                1000: "1,000",
                                2000: "2,000",
                                3000: "3,000",
                                4000: "4,000",
                                5000: "5,000"
                            },
                            value = 10
                        ),
                        html.Hr(),
                        html.H5("Score"),
                        html.P("A score of 1.02 indicates a predicted 2% increase in Revenue by optimizing the marketing budget."),
                        daq.LEDDisplay(
                            id = "digital-score", 
                            value = 1.09,
                            color="#92e0d3",
                            backgroundColor="#1e2130",
                            size=50,
                        ),
                        html.Br(),
                        # html.Button("Download Media Spend", id="btn"), dcc.Download(id="download")
                    ],
                    width = 3,
                    style={'margin':'10px'}
                ),
                dbc.Col([
                    # dcc.Graph(id='graph-slider'),
                    dcc.Graph(id='spend-graph',style={'width': '100%', 'height': '65vh'}),
                    html.Br(),
                    dcc.Graph(figure=fig2,style={'width': '100%', 'height': '65vh'})
                    
                ],width = 8),
                dcc.Store(id='intermediate-value')
            ] 
        )
    ]
)


# In[89]:


@app.callback(
    Output('intermediate-value', 'data'), 
    Input('budget-size', 'value'))
def process_data(budget_size):
    
    budget_grid_df = generate_grid(budget_size)

    budget_optimized = optimize_budget(
        df          = X_train,
        grid        = budget_grid_df, 
        media_spend = media_spend_df,
        verbose     = True
    )    

    budget_optimized_json = budget_optimized.copy()

    budget_optimized_json['rebalancing_coef'] = budget_optimized_json['rebalancing_coef'][0].to_json()

    budget_optimized_json['media_spend_rebal'] = budget_optimized_json['media_spend_rebal'].to_json()

    budget_optimized_json = json.dumps(budget_optimized_json)

    return budget_optimized_json

@app.callback(
    Output('spend-graph', 'figure'),
    Input('intermediate-value', 'data'))
def update_figure(budget_optimized_json):

    budget_optimized = json.loads(budget_optimized_json)
    
    fig = pd.read_json(budget_optimized['media_spend_rebal']) \
        .melt(
            id_vars='media'
        ) \
        .pipe(
            px.bar,
            x = 'variable', 
            y = 'value',
            color = 'media', 
            barmode = 'group',
            template = 'plotly_dark',
            title= "Spend: Current Plan vs Optimized"
           
        ) 

    return fig




@app.callback(
    Output('digital-score', component_property='value'),
    Input('intermediate-value', 'data'))
def update_digital(budget_optimized_json):

    budget_optimized = json.loads(budget_optimized_json)

    return np.round(float(budget_optimized['score']), 2)

# Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# In[91]:


app.run_server(port=4000)


# In[93]:


if __name__ == '__main__':
    app.run_server(debug=True,port=4000)


# In[ ]:





# In[ ]:




