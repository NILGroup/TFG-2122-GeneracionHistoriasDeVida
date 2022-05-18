import dash
from dash import dcc
from dash import html


import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from colour import Color
from textwrap import dedent as d
import json

import math
from scripts.Graph import Graph
from scripts.TriplesGenerator import TriplesGenerator


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Life Story Book"

TAGS = ['timeless','late chilhood','adolescence','early adulthood','late adulthood']
ACCOUNT = "A0001"
FILEURL = "data/example.json"

def read_file(fileUrl):
    with open(fileUrl) as f:
        json_data = json.loads(f.read())
    return json_data

def network_graph(fileUrl, tags, AccountToSearch):


    tg = TriplesGenerator(fileUrl)
    edges = tg.getData()


    #######     FILTERING       #########

    nodesSet = set()
    for index,item in edges.iterrows():
        
        if(item['stage'] not in tags):
            edges.drop(axis=0, index=index, inplace=True)
            continue

        nodesSet.add(item['source'])
        nodesSet.add(item['target'])

    # to define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.append('Elisa')
    shells.append(shell1)
    shell2=[]
    for ele in nodesSet:
        if ele!='Elisa':
            shell2.append(ele)
    shells.append(shell2)
    
    G = nx.from_pandas_edgelist(edges, 'source', 'target', ['source','target','relation','stage','themes'], create_using=nx.MultiDiGraph())


    # pos = nx.layout.circular_layout(G)
    # nx.layout.shell_layout only works for more than 3 nodes
    if len(shell2)>1:
        pos = nx.drawing.layout.shell_layout(G, shells)
    else:
        pos = nx.drawing.layout.spring_layout(G)

    pos = nx.drawing.spring_layout(G,dim=3, k=2/(G.number_of_nodes()), seed=18)

    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])


    if len(shell2)==0:
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

        node_trace = go.Scatter(x=tuple([1]), y=tuple([1]), text=tuple([str(AccountToSearch)]), textposition="bottom center",
                                mode='markers+text',
                                marker={'size': 50, 'color': 'LightSkyBlue'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
                                mode='markers',
                                marker={'size': 50, 'color': 'LightSkyBlue'},
                                opacity=0)
        traceRecode.append(node_trace1)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure


    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
    ############################################################################################################################################################
    

    index = 0
    for edge in G.edges:
        x0, y0,z0 = G.nodes[edge[0]]['pos']
        x1, y1,z1 = G.nodes[edge[1]]['pos']
        trace = go.Scatter3d(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),z=tuple([z0, z1, None]),
                           mode='lines',
                           marker=dict(color='dimgray'),
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1
    ###############################################################################################################################################################
    
    
    sizes = [d[1] for d in G.degree]
    maxis = max(sizes)
    sizes = [((math.log(s)*10)+20) for s in sizes]
    text_sizes = [int(((math.log(s)*10)+20)/5) for s in sizes]
    

    node_trace = go.Scatter3d(x=[], y=[], z=[], hovertext=[], text=[], mode='markers+text', textposition="middle center",
                            hoverinfo="text", marker={'size': sizes, 'sizemin': 20,'color': 'lightcoral'},textfont={
                                'size': text_sizes
                            })


    node_trace['text'] += tuple(G.nodes())
    index = 0
    for node in G.nodes():
        x, y, z = G.nodes[node]['pos']
        hovertext = "Name: " + str(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['z'] += tuple([z])
        node_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(node_trace)
    ################################################################################################################################################################
    middle_hover_trace = go.Scatter3d(x=[], y=[], z=[], hovertext=[], mode='markers+text', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0, z0 = G.nodes[edge[0]]['pos']
        x1, y1, z1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['source']) + "<br>" + "To: " + str(
            G.edges[edge]['target']) + "<br>" + "Relation: " + str(
            G.edges[edge]['relation']) + "<br>" + "Stage: " + str(G.edges[edge]['stage'] +"<br>"+"Themes: " + str(G.edges[edge]['themes']))
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['z'] += tuple([(z0 + z1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    #################################################################################################################################################################
    figure = {
        "data": traceRecode,
        
        "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False, hovermode='closest',
                            margin={'b': 20, 'l': 20, 'r': 20, 't': 20},
                            scene=go.layout.Scene(
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                zaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},),
                                height=800,
                                clickmode='event+select',
                                #  annotations=[
                                #      dict(
                                #          ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                #          ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2,
                                #          az=(G.nodes[edge[0]]['pos'][2] + G.nodes[edge[1]]['pos'][2]) / 2, axref='x', ayref='y', azref="z",
                                #          x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                #          y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, 
                                #          z=(G.nodes[edge[1]]['pos'][2] * 3 + G.nodes[edge[0]]['pos'][2]) / 4, xref='x', yref='y', zref="z",
                                #          showarrow=True,
                                #          arrowhead=3,
                                #          arrowsize=4,
                                #          arrowwidth=1,
                                #          opacity=1
                                # ) for edge in G.edges]
                            )}
    return figure
######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    #########################Title
    html.Div([html.H1("Life Story Book")],
             className="row",
             style={'textAlign': "center"}),
    #############################################################################################define the row
    html.Div(
        className="row",
        children=[
            ##############################################left side two input components
            html.Div(
                className="two columns",
                children=[
                   
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **Account To Search**

                            Input the account to visualize.
                            """)),
                            dcc.Input(id="input1", type="text", placeholder="Account"),
                            html.Div(id="output")
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),

            ############################################middle graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(FILEURL, TAGS, ACCOUNT))],
            ),

            #########################################right side two output component
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**

                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**

                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])

###################################callback for left side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])

def update_output(value,input1):
    TAGS = value
    ACCOUNT = input1
    return network_graph(FILEURL, value, input1)

    # to update the global variable of TAGS and ACCOUNT
################################callback for right side components
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



if __name__ == '__main__':
    app.run_server(debug=True)
