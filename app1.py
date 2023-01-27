
import pandas as pd
import numpy as np 
import plotly.graph_objects as go 
import plotly as plt
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import dash_auth
import pickle
from dash import Dash, dcc, html, Input, Output



raw_symbols = SymbolValidator().values
symbols = []
for i in np.arange(0,len(raw_symbols), 3):
    symbols.append(raw_symbols[i+2])


with open('names.pkl', 'rb') as f:
    names = pickle.load(f)


with open('df_lists.pkl', 'rb') as f:
    df_lists = pickle.load(f)


plots_names = ['weakness', 'std', 'std_average', 'std_weak', 'p_average', 'p_repitition_average', 'p_median','p_median_all', 'p_median_average','p_range', 'p_range_average']
colors = {'background': '#111111', 'text': '#7FDBFF'}

USERNAME_PASSWORD_PAIRS = [['talal', 'ghannam']]
app = Dash()
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server
app.layout = html.Div( children=[
    html.H4('Dieharder Tests Plots'),
    html.P('Chose Plot Type'),
    dcc.RadioItems(plots_names, plots_names[0], id="plot-picker", ),
    html.P('Plot Description'),
    dcc.Markdown(id='plot-explain',  link_target="_blank", ),
    html.P("Filter by test:"),
    dcc.Dropdown(names, names[0], id="test-picker", multi = True), 
    dcc.Graph(id="plot",  style={'width':'85%', 'float': 'left','height': '70vh','display':'inline-block'}),
    
                    ])
                    

@app.callback(
    Output("plot", "figure"), 
    [Input("plot-picker", "value"), Input("test-picker", "value")])
def update_bar_chart(plot_picker, picker_test):
    i=0
    if plot_picker == 'weakness':              
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace = go.Bar(x=df.test_name, y = df.weak_rate, name = '{}, #rounds: {}'.format(test,n_rounds))
            data.append(trace)
        
        layout = go.Layout(title = 'Fraction of weak and failed results per each Dieharder test')
        fig = go.Figure(data, layout)
        fig.update_yaxes(title_text='Failed/weak fractions')
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    
    elif plot_picker == 'std':
        data = []
        
        for test in picker_test:
            i +=1
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace = go.Scatter(x = df['test_name'], y = df['std'], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), marker =dict(symbol=symbols[i], opacity=0.4), line_width = 1)
            data.append(trace)
        
        layout = go.Layout(title = "The standard deviation of the p_values for each Dieharder test")
        layout.xaxis.dtick
        fig = go.Figure(data, layout)
        fig.update_yaxes(title_text='Std', range = [0,1])
        fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        fig.add_hline(y = 0.2886, line_color = 'black', line_dash = 'dash', annotation_text = '0.288-line', annotation_position="top right", )

        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    
    elif plot_picker == 'std_weak':
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            threshold = 0.15
            #df_weak = df[df['std'] <= threshold]
            df_weak = df['std'].apply(lambda x: x if x<= threshold else 0)
            trace = go.Bar(x = df['test_name'], y = df_weak, name  = '{}, #rounds: {}'.format(test,n_rounds))
            data.append(trace)
    
        layout = go.Layout(title = 'Dieharder tests that have their standard deviation <= {}.'.format(threshold), )
        layout.xaxis.dtick
        fig = go.Figure(data, layout)
        fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        fig.update_yaxes(title_text='Fractions of rounds with weak std')
        #fig.update_xaxes(dtick=df.test_name,  tickmode = 'linear')
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    
    
    elif plot_picker == 'std_average':
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            df_group = df[['test_name', 'std']].groupby('test_name').mean()
            df_group['tests_count'] = df.test_name.value_counts()
            trace = go.Scatter(x = df_group.index, y = df_group['std'], mode = 'lines+markers', text = df_group['std'], name  = '{}, #rounds: {}'.format(test,n_rounds),
            marker=dict(size=df_group.tests_count,), line_width = 1)
            data.append(trace)
        
        layout = go.Layout(title="Std averaged over each test's repetitions. <br>Circles indicate number of repitions for each test")
        fig = go.Figure(data, layout)
        fig.update_yaxes(title_text='Std', range = [0, 1])
        fig.add_hline(y = 0.2886, line_color = 'black', line_dash = 'dash', annotation_text = '0.288-line', annotation_position="top right", )
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        #fig.add_hline(np.mean(df_group['std']), line_color = 'red', line_dash = 'dash', annotation_text = 'std mean', annotation_position="top right")
        #fig.add_hline(np.median(df_group['std']), line_color = 'green', line_dash = 'dash', annotation_text = 'std median', annotation_position="top left")
        return fig
    
    
    elif plot_picker == 'p_average':
        data_1 = []
        data_2 = []
        fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.5)
        cols = plt.colors.DEFAULT_PLOTLY_COLORS
        i=0
        for test in picker_test:
            i+=1
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace_1 = go.Scatter(x = df['name_index'][:55], y = df['p_mean'][:55], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), line_color=cols[i], line_width = 1)
            trace_2 = go.Scatter(x = df['name_index'][55:], y = df['p_mean'][55:], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), line_color= cols[i], line_width = 1)
            data_1.append(trace_1)
            data_2.append(trace_2)
            fig.append_trace(trace_1, row=1, col=1)
            fig.append_trace(trace_2, row=2, col=1)
        layout = go.Layout(title = 'Averag p_values for each Dieharder test.' ) #<br> min = {}, max = {}'.format(df['p_mean'].min(), df['p_mean'].max())
        layout.xaxis.dtick
        fig.update_layout(layout)
        fig.update_yaxes(title_text='P_value', range = [0,1])
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.update_xaxes(categoryorder='array', categoryarray= df.name_index)
        return fig
    
    elif plot_picker == 'p_repitition_average':
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            df_group = df[['test_name', 'p_mean']].groupby('test_name').mean()
            df_group['tests_count'] = df.test_name.value_counts()
            trace = go.Scatter(x = df_group.index, y = df_group['p_mean'], mode = 'lines+markers', text = df_group['p_mean'], name  = '{}, #rounds: {}'.format(test,n_rounds),
            marker=dict(size=df_group.tests_count,), line_width = 1)
            data.append(trace)

        layout = go.Layout(title="Average p_values averaged over each Dieharder test's repititions. <br>The circles are proportional to the number of repitions for each test")
        fig = go.Figure(data, layout)
        fig.add_hline(y = 0.5, line_color = 'black', line_dash = 'dash', annotation_text = '0.5-line', annotation_position="top right", )
        fig.update_yaxes(title_text='P_value', range = [0, 1])
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        #fig.add_hline(np.mean(df_group['std']), line_color = 'red', line_dash = 'dash', annotation_text = 'median mean', annotation_position="top right")
        #fig.add_hline(np.median(df_group['std']), line_color = 'green', line_dash = 'dash', annotation_text = 'std median', annotation_position="top left")
        return fig
    
    
    elif plot_picker == 'p_median':
        data = []
        for test in picker_test:
            i+=1
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace = go.Scatter(x = df['test_name'], y = df['p_median'], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), marker =dict(symbol=symbols[i]), line_width = 1)
            data.append(trace)
        
        layout = go.Layout(title="P_values's median per Dieharder tests. ") #<br> min = {}, max = {}".format(df['p_median'].min(), df['p_median'].max())
        fig = go.Figure(data, layout)
        fig.add_hline(y = 0.5, line_color = 'black', line_dash = 'dash', annotation_text = '0.5-line', annotation_position="top right", )
        fig.update_yaxes(title_text='Median of p_values', range =[0, df['p_median'].max()+0.1])
        #fig.add_hline(np.mean(df['p_median'] ), line_color = 'red', line_dash = 'dash', annotation_text = 'p_median mean', annotation_position="top right")
        #fig.add_hline(np.median(df['p_median'] ), line_color = 'green', line_dash = 'dash', annotation_text = 'p_median median', annotation_position="top left")
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    
    elif plot_picker == 'p_median_all':
        data_1 = []
        data_2 = []
        fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.5)
        cols = plt.colors.DEFAULT_PLOTLY_COLORS
        i=0
        for test in picker_test:
            i+=1
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace_1 = go.Scatter(x = df['name_index'][:55], y = df['p_median'][:55], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), line_color=cols[i], line_width = 1)
            trace_2 = go.Scatter(x = df['name_index'][55:], y = df['p_median'][55:], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds), line_color= cols[i], line_width = 1)
            data_1.append(trace_1)
            data_2.append(trace_2)
            fig.append_trace(trace_1, row=1, col=1)
            fig.append_trace(trace_2, row=2, col=1)
        layout = go.Layout(title = 'P_values medians for all Dieharder test.' ) #<br> min = {}, max = {}'.format(df['p_mean'].min(), df['p_mean'].max())
        layout.xaxis.dtick
        fig.update_layout(layout)
        fig.update_yaxes(title_text='Median of P_values', range = [0,1])
        #fig.update_xaxes(categoryorder='array', categoryarray= df.name_index)
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    
    elif plot_picker == 'p_median_average':
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            df_group = df[['test_name', 'p_median']].groupby('test_name').mean()
            df_group['tests_count'] = df.test_name.value_counts()
            trace = go.Scatter(x = df_group.index, y = df_group['p_median'], mode = 'lines+markers', text = df_group['p_median'], name  = '{}, #rounds: {}'.format(test,n_rounds),
            marker=dict(size=df_group.tests_count,), line_width = 1)
            data.append(trace)
        
        layout = go.Layout(title="P_values' medians averaged over each test's repitions. <br>The circles are proportional to the number of repitions for each test")
        fig = go.Figure(data, layout)
        fig.add_hline(y = 0.5, line_color = 'black', line_dash = 'dash', annotation_text = '0.5-line', annotation_position="top right", )
        fig.update_yaxes(title_text='Median of p_values', range = [0, 1])
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.add_hline(np.mean(df_group['std']), line_color = 'red', line_dash = 'dash', annotation_text = 'median mean', annotation_position="top right")
        #fig.add_hline(np.median(df_group['std']), line_color = 'green', line_dash = 'dash', annotation_text = 'std median', annotation_position="top left")
        #fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        return fig
    
    elif plot_picker == 'p_range':
        data = []
        for test in picker_test:
            i+=1
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            trace = go.Scatter(x = df['test_name'], y = df['range'], mode = 'lines+markers', name  = '{}, #rounds: {}'.format(test,n_rounds),marker =dict(symbol=symbols[i]), line_width = 1)
            data.append(trace)
            
        layout = go.Layout(title="P_values' ranges per each Dieharder tests.")  #<br> min = {}, max = {}".format(df['range'].min(), df['range'].max())
        fig = go.Figure(data, layout)
        fig.update_yaxes(title_text='Range of p_value', range =[0, df['range'].max()+0.1])
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.add_hline(np.mean(df['p_median'] ), line_color = 'red', line_dash = 'dash', annotation_text = 'p_median mean', annotation_position="top right")
        #fig.add_hline(np.median(df['p_median'] ), line_color = 'green', line_dash = 'dash', annotation_text = 'p_median median', annotation_position="top left")
        return fig
    
    elif plot_picker == 'p_range_average':
        
        data = []
        for test in picker_test:
            df = df_lists[test]
            p_value = [x for x in df.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            df_group = df[['test_name', 'std','range']].groupby('test_name').mean()
            df_group['tests_count'] = df.test_name.value_counts()
            #df_group['std'] = df.std.mean()
            trace = go.Scatter(x = df_group.index, y = df_group['range'], mode = 'lines+markers', text = df_group['range'], name  = '{}, #rounds: {}'.format(test,n_rounds),
            marker=dict(size=df_group.tests_count,), line_width = 1)
            data.append(trace)
        
        layout = go.Layout(title="P_values' ranges averaged over each test's repition. <br>The circles are proportional to thenumber of repitions for each test")
        fig = go.Figure(data, layout)
        fig.update_yaxes(title_text='Range of p_values', range = [0, 1])
        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        #fig.add_hline(np.mean(df_group['std']), line_color = 'red', line_dash = 'dash', annotation_text = 'median mean', annotation_position="top right")
        #fig.add_hline(np.median(df_group['std']), line_color = 'green', line_dash = 'dash', annotation_text = 'std median', annotation_position="top left")
        #fig.update_xaxes(categoryorder='array', categoryarray= df.test_name)
        return fig

@app.callback(Output('plot-explain', 'children'), [Input("plot-picker", "value")])
def update_chart_info(picker_test):
    if picker_test == 'weakness':
        text = "This plot counts how many weak/fail rounds are there for each test and divides it by the total number of rounds."
    elif picker_test == 'std':
        text = "This plot claculates the standard deviation of the p_values of each test. \nThe expected value for uniform distribution is std = 1/sqrt(12) = 0.2886..."
    elif picker_test == 'std_average':
        text = "This plot claculates the standard deviation of the p_values averaged over each test's repitition."
    elif picker_test == 'std_weak':
        text = "This plot returns the fraction of rounds for each Dieharder tests that have their standard deviations less than a specific threshold."
    elif picker_test == 'p_average':
        text = "This plot returns the average p_values for all Dieharder tests on the x-axis. \nFor a high number of rounds, the expected value for a uniform distribution is 0.5."
    elif picker_test == 'p_repitition_average':
        text = "This plot returns the average p_values over tests's rounds and then averages the values over each Dieharder test's repitition."
    elif picker_test == 'p_median':
        text = "This plot returns the median of the p_values of each Dieharder test. \nFor a high number of rounds, the expected value for a uniform distribution is 0.5."
    elif picker_test == 'p_median_all':
        text = "This plot returns the median of the p_values for all Dieharder tests on the x-axis."
    elif picker_test == 'p_median_average':
        text = "This plot returns the median of the p_values averaged over each Dieharder test's repitition."
    elif picker_test == 'p_range':
        text = "This plot returns the range of the p_values for each Dieharder test. \nFor a high number of rounds, the expected value for a uniform distribution is 1."
    elif picker_test == 'p_range_average':
        text = "This plot returns the range of the p_values averaged over each Dieharder test's repitition."
    else:
        return ""
    return [text]
    
app.run_server(debug=True, use_reloader=False )
           





