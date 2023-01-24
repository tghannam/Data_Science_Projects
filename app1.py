import pandas as pd 
import numpy as np 
import plotly.graph_objects as go 
import plotly as plt
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.validators.scatter.marker import SymbolValidator
from dash import dash_table
import dash_auth
raw_symbols = SymbolValidator().values
symbols = []
for i in np.arange(0,len(raw_symbols), 3):
    symbols.append(raw_symbols[i+2])


xls = pd.ExcelFile('NEW COMBINED_test_results_221214_.xlsx')
df_32_64_4 = pd.read_excel(xls, '32-64-4')
df_32_64_2 = pd.read_excel(xls, '32-64-2')
df_32_64_1 = pd.read_excel(xls, '32-64-1')
df_32_8_2 = pd.read_excel(xls, '32-8-2')
df_32_8_1 = pd.read_excel(xls, '32-8-1')

def assessment_value(x):
    if x >0.995 or x <0.005:
        return 'WEAK'
    elif x >0.999999 or x <10^-61:
        return 'FAIL'
    else:
        return 'PASS'


excel_32_8_32 = pd.ExcelFile('32_8_32_myedits.xlsx')
df_32_8_32 = pd.read_excel(excel_32_8_32)
df_32_8_32['test_name'] = df_32_8_32['test_name_ntup']
df_32_8_32['test_name'] = df_32_64_4['test_name']


pvalue = [x for x in df_32_8_32.columns if x.startswith('pva')]
assessment = ['assessment']
range = list(np.arange(1,190))
for i in range:
    assessment.append(f'assessment.{i}')
assessment;
df_dict_32_8_32 = {}
p_a = list(zip(pvalue, assessment))
for i in np.arange(len(df_32_8_32)):

    for x in p_a:
        df_32_8_32[x[1]] = df_32_8_32[x[0]]
        df_32_8_32[x[1]] = df_32_8_32[x[1]].apply(assessment_value)
    df_32_8_32 = df_32_8_32[(df_32_8_32['test_name'] != 'diehard_opso') &  (df_32_8_32['test_name'] !=  'diehard_oqso') & (df_32_8_32['test_name'] !=  'diehard_dna') & (df_32_8_32['test_name'] != 'diehard_sums')]
    df_dict_32_8_32['32_8_32'] = df_32_8_32


DNR = df_32_8_1[df_32_8_1['assessment'] == 'DNR']['test_name'].to_list()
assessment = [x for x in df_32_64_4.columns if x.startswith('asse')]
p_value = [x for x in df_32_64_4.columns if x.startswith('pva')]
RNG_Name =['mt19937',	'ranlux',	'ranlxd',	'uvag',	'mrg',	'knuthran',	'knuthran2002',	'R_knuth_taocp2',	'ranlxd2',	'ranmar',	'R_knuth_taocp',	'tt800',	'R_wichmann_hill',	'ran1',]


excel = pd.ExcelFile('WIP-RESULTS-ALL-RNGs 221215_ta.xlsx')
df_all = pd.read_excel(excel, 'PVALUE-Analysis')


df = pd.DataFrame()
df_dict_rng = {}
value  = list(zip(p_value, assessment))

for name in RNG_Name:
    for i in np.arange(len(df_all)):
        if df_all['RNG Name & Test Name'].iloc[i] == name:
            df= df_all.iloc[i+1:i+115, :11]
            df.columns = ['test_name_ntup', 'pvalue', 'pvalue.1',	'pvalue.2',	'pvalue.3',	'pvalue.4',	'pvalue.5',	'pvalue.6',	'pvalue.7',	'pvalue.8',	'pvalue.9',	]
            df['test_name'] = df['test_name_ntup']
            df.reset_index(inplace=True)
            df.drop('index', inplace=True, axis = 1)
            df['test_name'] = df_32_8_1['test_name']
            for x in value:
                df[x[1]] = df[x[0]]
                df[x[1]] = df[x[1]].apply(assessment_value)
            df = df[(df['test_name'] != 'diehard_opso') &  (df['test_name'] !=  'diehard_oqso') & (df['test_name'] !=  'diehard_dna') & (df['test_name'] != 'diehard_sums')]
            df_dict_rng[name] = df


import pandas as pd

sheets_dict = pd.read_excel('NEW COMBINED_test_results_221214.xlsx', sheet_name=None)

all_sheets = []
for name, sheet in sheets_dict.items():
    sheet['sheet'] = name
    sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
    all_sheets.append(sheet)

'''full_table = pd.concat(all_sheets)
full_table.reset_index(inplace=True, drop=True)'''


xl = pd.ExcelFile('NEW COMBINED_test_results_221214_no_summary.xlsx')
sheet_names = xl.sheet_names

sheets_dict = pd.read_excel('NEW COMBINED_test_results_221214_no_summary.xlsx', sheet_name=None)
df_dict = {}
for i in np.arange(len(sheet_names)):
    df_dict[sheet_names[i]] = list(sheets_dict.values())[i]

df['index_'] = df.index.values
df['name_index'] = df['index_']
for i in np.arange(len(df)):
    df['name_index'].iloc[i] = df['test_name'].iloc[i]+'_'+str(df['index_'].iloc[i])


class Dieharder_plot():
    def __init__(self, file_path):
        
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        df_dict = {}
        for i in np.arange(len(sheet_names)):
            df_dict[sheet_names[i]] = list(sheets_dict.values())[i]
        #df_dict.pop('SUMMARY')    
        df_dict  = df_dict|df_dict_rng
        df_dict = df_dict|df_dict_32_8_32
        self.df_dict = df_dict
        '''names = []
        for d in df_list:
            n = [x for x in globals() if globals()[x] is d][0]
            names.append(n)
        self.names = names
        '''
    def data_frames(self, ):
        
        #df_list = self.df_list
        df_lists = {}
        name_list = []
        df_dict = self.df_dict
        for name_, df_ in df_dict.items():
            #name = [x for x in globals() if globals()[x] is df_][0]
            name = name_
            assessment = [x for x in df_.columns if x.startswith('asse')]
            p_value = [x for x in df_.columns if x.startswith('pva')]
            n_rounds = len(p_value)
            df_ = df_[df_['assessment'] != 'DNR']
            index = list(set(df_.test_name))
            try:
                df_['weak_rate'] = df_.apply(pd.Series.value_counts, axis=1)['WEAK'].fillna(0)/n_rounds;
            except:
                print(f'No Weak: {name}')
            try:
                df_['failed_rate'] = df_.apply(pd.Series.value_counts, axis=1)['FAILED'].fillna(0)/n_rounds;
            except:
                print(f'No Failed: {name}')
            df_['std'] = df_['pvalue']*0
            for i in np.arange(len(df_)):
                std = np.std(df_[p_value].iloc[i]);
                df_['std'].iloc[i] = std;
            df_['p_mean'] = df_[p_value].apply(pd.Series.mean, axis=1)
            df_['p_median'] = df_[p_value].apply(pd.Series.median, axis = 1)
            #df_['std'] = df_[p_value].apply(pd.Series.std, axis = 1)
            df_['range'] = df_.weak_rate
            for i in np.arange(len(df_)):
                max= df_[p_value].iloc[i].max()
                min = df_[p_value].iloc[i].min()
                df_['range'].iloc[i] = max-min
            #df_['index_'] = df_.index.values
            df_['name_index'] = df_['p_mean']
            for i in np.arange(len(df_)):
                df_['name_index'].iloc[i] = df_['test_name'].iloc[i]+'_'+str(i)
            df_lists[name] = df_
            name_list.append(name)
            self.names = name_list
        self.index = index
        self.df_lists = df_lists
        self.n_rounds = n_rounds
    
    #style={'width': '100vh', 'height': '70vh'}
    #children=html.H4() ,
    #style={'whiteSpace': 'pre-line'}
    #html.Div([html.Pre(id='hover-data', style ={'paddingTop':35})], style={'width':'20%'})
    #html.Div([dcc.Graph(id='hover-data', style ={'paddingTop':35, 'float':'right'})], style={'width':'20%'}),
    #   '''selected_columns=[],  # ids of columns that user selects                                  selected_rows=[], '''
    def plots(self,):    
        
        df_lists = self.df_lists
        plots_names = ['weakness', 'std', 'std_average', 'std_weak', 'p_average', 'p_repitition_average', 'p_median','p_median_all', 'p_median_average','p_range', 'p_range_average']
        colors = {'background': '#111111', 'text': '#7FDBFF'}
        from dash import Dash, dcc, html, Input, Output, State
        names = self.names
       
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
            
        self.app = app
           


file_path = 'NEW COMBINED_test_results_221214_no_summary.xlsx'


dieharder = Dieharder_plot(file_path)


pd.options.mode.chained_assignment = None 
dieharder.data_frames()
df = dieharder.df_lists['32-64-4'];


dieharder.plots()
dieharder.app.run_server(debug=True, use_reloader=False );     


