import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from LSTM import get_days, get_pred, get_actual
import dash
from prediction import get_closing_price, get_days_2
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


app = dash.Dash(__name__)

# df = pd.read_csv("intro_bees.csv", sep=',')
# df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
# df.reset_index(inplace=True)
# print(df[:5])



#App Layout

app.layout = html.Div([
    html.H1("Web Application Dashboard with Dash", style={'text-align' : 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "Decision Tree Regressor", "value": 1},
                     {"label": "Long-Shor Term Memory", "value": 2},
                     {"label": "SVM", "value": 3},
                     {"label": "Linear Regressor", "value": 4}],
                 multi=False,
                 value=1,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})

])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components


@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    # dff = df.copy()
    # dff = dff[dff["Year"] == option_slctd]
    # dff = dff[dff["Affected by"] == "Varroa_mites"]


    print(get_days_2())

    if option_slctd == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=get_days_2(), y=get_closing_price(),
                                 mode='lines',
                                 name='lines'))



    elif option_slctd == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=get_days(), y=get_actual(),
                                 mode='lines',
                                 name='lines'))

        fig.add_trace(go.Scatter(x=get_days(), y=get_pred(),
                                 mode='lines',
                                 name='lines'))



    return container, fig

if __name__ == '__main__':
    app.run_server(debug=True)