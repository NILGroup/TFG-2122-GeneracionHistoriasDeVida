import math
from multiprocessing.sharedctypes import Value
from pydoc import classname
from typing import OrderedDict
import dash
from dash import dcc
from dash import html
import numpy as np


import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from colour import Color
from textwrap import dedent as d
import json



from scripts.Graph import Graph
from scripts.Model import Model
from scripts.TripleList import TripleList
from scripts.Triples2Sentence import Triples2Sentence
from scripts.TriplesClustering import TriplesClustering
from scripts.TriplesGenerator import TriplesGenerator

import dash_bootstrap_components as dbc
from dash.dependencies import ClientsideFunction, Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"])
app.title = "Generación de historias de vida"

INITIAL_TAGS = []
ACCOUNT = "A0001"
FILEURL = "data/example.json"
SELECTED_TAGS = []  
ORDER_TAGS = []

TRIPLES = TriplesGenerator(FILEURL)

model = Model() 

model.load_model("models/T5_webnlg_5000samples_2epochs")


def printAllText(triplesClusters, do_sample = False, num_beams = 5, no_repeat_ngram_size = 2, min_length = 0, max_length = 500, top_k = 4, top_p = 0.30, temperature = 0.8, penalty = 0.5, num_return_sequences = 1, early_stopping = False):
    final_text = ""
    
    for item in triplesClusters:
        if(item=='input'):
            cluster_set = set(triplesClusters[item]['cluster'])

            for c in cluster_set:             
                index = np.where(triplesClusters[item]['cluster']==c)[0]
                input =  []
                for i in index:
                    input.append(triplesClusters[item]['input'][i])
                
                triple2sen = Triples2Sentence(input)
                prompt = triple2sen.getText()

                inputs_id = model.encode(prompt=prompt)
                outputs = model.generateText(inputs_id , do_sample, num_beams, no_repeat_ngram_size,len(input)*min_length, max_length, top_k , top_p, temperature, penalty , num_return_sequences, early_stopping)

                for output in outputs:
                    text = model.decode(output)
                    text = text.replace('<pad>', '')
                    text = text.replace('</s>', '')
                    final_text += text

        else:
            final_text+=printAllText(triplesClusters[item],do_sample, num_beams, no_repeat_ngram_size,min_length, max_length, top_k , top_p, temperature, penalty , num_return_sequences, early_stopping)
    return final_text

def filteredTriplesByTags(triplesTree, themesSelection):
    if(len(themesSelection)==0):
        return []
    else:
        diccionario = {}
        for t in themesSelection:
            if 'tag' in triplesTree[t]:
                lista = filteredTriplesByTags(triplesTree[t]['tag'], themesSelection[t])
                if(lista==[]):
                    diccionario[t] = triplesTree[t]
                else:
                    diccionario[t] = lista
            else:        
                diccionario[t] = triplesTree[t]

        return diccionario

def clusteringByTags(themesSelection):

    triplesClustersList = {}
    for tag in themesSelection:

        if tag == 'input':
            clustering = TriplesClustering(themesSelection)
            clustering.genClusters()
            return {'input' : clustering.getTriples()}
               
        triplesClustersList[tag] = clusteringByTags(themesSelection[tag])
    
    return triplesClustersList

def listTags2Tree(list_tags):
    list_dict_tags = []
    for lt in list_tags:
        list_dict_tags.append(lt.split('_'))

    dict_tags ={}
    for ldt in list_dict_tags:
        
        pred = dict_tags
        for tag in ldt:
            if(tag not in pred): #Si no estaba la añado
                pred[tag] = {}
                
            pred=pred[tag]

    return dict_tags

def network_graph(fileurl,tags, AccountToSearch):
    if(len(tags)==0):
        figure = {
            "layout": go.Layout(title='Visualización interactiva', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure

    edges = TRIPLES.getData().copy()


    #######     FILTERING       #########

    nodesSet = set()
    #tags_selected = listTags2Tree(tags)

    stages = []
    for stage in tags:
        stages.append(stage.split('_')[0])


    for index,item in edges.iterrows():

        conservar = 0
        
        if(len(tags)==0):
            conservar = 1
        
        j=0    
        while(j<len(tags) and not conservar):
            theme = tags[j]
            count = sum(1 for s in tags if theme in s)
            if(count==1): # If only appears 1 time
                
                if(item['stage'] == theme.split('_')[0]): # If the stage is correct
                    if(theme in stages):
                        conservar = 1
                    else:    
                        # If there arent themes in tags selected
                        allthemes = 1
                        for i in range(0, len(theme.split('_'))-1):
                            if(len(item['themes']) <= i or item['themes'][i]!=theme.split('_')[i+1]):   
                                allthemes = 0
                            else:
                                
                                allthemes &= 1
                        conservar = allthemes
                        
            j = j+1
           
        if(not conservar):
            edges.drop(axis=0, index=index, inplace=True)
            
        else:            
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


    if len(shell2)<1:
        pos = nx.drawing.layout.shell_layout(G, shells)
    else:
        pos = nx.drawing.layout.spring_layout(G,k=2/(G.number_of_nodes()), seed=18)



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
            "layout": go.Layout(title='Visualización interactiva', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure


    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
    ############################################################################################################################################################
    
    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           marker=dict(color='darkgray'),
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1
    ###############################################################################################################################################################
    sizes = [d[1] for d in G.degree]
    maxis = max(sizes)
    sizes = [((math.log(s)*10)+20) for s in sizes]
    text_sizes = [int(((math.log(s)*10)+20)/5) for s in sizes]
    
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text',textposition="middle center",
                            hoverinfo="text", 
                            marker={'size': sizes, 'color': 'lightcoral'},
                            textfont={
                                'size': text_sizes
                            })

    node_trace['text'] += tuple(G.nodes())
    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "text: " + str(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(node_trace)
    ################################################################################################################################################################
    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['source']) + "<br>" + "To: " + str(
            G.edges[edge]['target']) + "<br>" + "Relation: " + str(
            G.edges[edge]['relation']) + "<br>" + "Stage: " + str(G.edges[edge]['stage'] +"<br>"+"Themes: " + str(G.edges[edge]['themes']))
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    #################################################################################################################################################################
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Visualización Interactiva', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=4,
                                    arrowwidth=1,
                                    opacity=1
                                ) for edge in G.edges]
                            )}
    return figure

def recursivetags(name, tags, pad, before_id, class_name):

    if(len(tags)==0):
        button = dbc.Button(name, n_clicks=0, class_name='theme tag', id=before_id)
        return dbc.Card(children = button, style={'marginLeft':str(pad)+'em'}), [before_id]
    else:
        tree = []
        list_id = []
    
        tree.append(dbc.Button(name, n_clicks=0, 
                        class_name=class_name, id=before_id))
        list_id.append(before_id)

        for t in tags:
            tree_children, ids_used = recursivetags(t, tags[t], pad+1, before_id+'_'+t,'theme tag')

            list_id.extend(list(ids_used))
            tree.append(tree_children)

        drag = html.Div(className = "draggable",children=tree)
        return dbc.Card(children = drag, style={'marginLeft':str(pad)+'em'}), list_id


def tags_network(fileUrl):


    tg = TriplesGenerator(fileUrl)
    triples = TripleList(data = tg.to_json())
    stages = triples.getTagStages()
    content = []
    list_id = []
    
    for stage in stages:
        tags =  triples.getTagTrees()[stage]
           
        c, l = recursivetags(stage, tags, 1, stage,'theme stage')
        content.append(c)
        list_id.extend(list(l))

    return content, list_id


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div(
    id="ppal_container",
    children=[

    #----------- titulo -----------#
    html.Div([html.H1(app.title)],
            className="row", 
            id="titulo"),

    #----------- contenido -----------#
    html.Div(
        className="row",
        children=[

            #----------- columna de tags -----------#
            html.Div(
                className="two columns",
                children=[
                html.Div(
                        className="twelve columns ",
                        style={'display':'grid'},
                        children=[
                            html.H2("Lista de tags", className="subtitle"),
                            
                            html.Div(
                                id="tags_header",
                                children=[
                                    html.Button('Todas', id="allTags", className="button options_button", n_clicks=0, style={'float':'left'}),
                                    html.Button('Ninguna', id="noneTags", className="button options_button", n_clicks=0, style={'float':'right'}),                              
                                    
                                ]
                            ),                               
                            
                            html.Div(
                                id="drag_container", 
                                className="container", 
                                children = tags_network(fileUrl=FILEURL)[0]
                            ),                            
                        ]
                    ),
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
            
            #----------- grafo de conocimiento y resultados de generación-----------#
            html.Div(
                className="eight columns",
                children=[
                    
                    dcc.Graph(id="my-graph",figure=network_graph(FILEURL, INITIAL_TAGS, ACCOUNT)),
                
                    html.Div(
                        className="form-outline",
                        children=[
                            html.Div(
                                className="Header",
                                style={'style':'inline-block', 'width':'100%'},
                                children=[
                                    html.H2('Salida',style={'float':'left'}),
                                    html.Button('Generar', id="generator", n_clicks=0, style={'float':'right'}, className="button")                                
                                ]
                            ),
                            dcc.Textarea(
                                className="form-control",
                                id="outputGenerator",
                                value="No output"
                            )
                        ]
                    
                    ),
                ]
            ),

            #----------- columna de información -----------#
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Información dinámica del nodo**

                            Información sobre el nodo al pasar el cursor del ratón por encima
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Información del nodo seleccionado**

                            Información del nodo seleccionado del grafo.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            ),
            html.Div("hola",id='hidden-div', style={'display':'none'})
        ]
    )
    
])


#----------CALLBACK----------
@app.callback(
    [Output('my-graph', 'figure'),
    [Output(i, 'class_name') for i in tags_network(FILEURL)[1]]],
    [Input(i, 'id') for i in tags_network(FILEURL)[1]],
    [Input(i, 'n_clicks') for i in tags_network(FILEURL)[1]]
)
def update_output(*args):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]    
    for changed in changed_id:
        
        changed = changed.split('.')[0]

        changed_clicks = args[args.index(changed) + int(len(args)/2)]

        if(changed_clicks%2==1): # No seleccionado
            if changed in SELECTED_TAGS:
                SELECTED_TAGS.remove(changed)

        else: # Seleccionado
            if changed not in SELECTED_TAGS:
                SELECTED_TAGS.append(changed)

    styles = []
    for i in range(0,int(len(args)/2)):
        theme_n_clicks = args[i+int(len(args)/2)]
        theme_name = args[i]
        if(theme_n_clicks%2==1): # No seleccionado
            if('_' in theme_name):
                styles.append('theme tag NotSelected')
            else:
                styles.append('theme stage NotSelected')

        else: # Seleccionado
            if('_' in theme_name):
                styles.append('theme tag')
            else:
                styles.append('theme stage')

      
    return network_graph(FILEURL, SELECTED_TAGS, 0), styles

@app.callback(
    Output('hover-data', 'children'),
    Input('my-graph', 'hoverData')
)
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('outputGenerator','value'),
    Input('generator','n_clicks'),Input("hidden-div", "className"),Input('outputGenerator','value')
)
def display_output_text(n_clicks,tags,text):
    if(len(SELECTED_TAGS)==0):
        return 'No output'
    data = TripleList(TRIPLES.to_json())
    triplesTree = data.triplesByTags()
    
    order_tags= []
    tags =tags.split(",")
    for tag in tags:
        if tag in SELECTED_TAGS:
            order_tags.append(tag)
    
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]

    if('generator.n_clicks' in changed_id):
        print("Generando...")
        triples_filtered = filteredTriplesByTags(triplesTree, listTags2Tree(order_tags))
        triplesClusters = clusteringByTags(triples_filtered)
        text = printAllText(triplesClusters)

    return text




@app.callback(
    [Output(i, 'n_clicks') for i in tags_network(FILEURL)[1]],
    Input('allTags','n_clicks'),
    Input('noneTags','n_clicks')
)
def allTasgNotSelected(n_clicks_all, n_clicks_none):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
    if 'noneTags.n_clicks' in changed_id:
        return list(np.ones(shape=(len(tags_network(FILEURL)[1]),)))
    else:
        return list(np.zeros(shape=(len(tags_network(FILEURL)[1]),)))



@app.callback(
    Output('click-data', 'children'),
    Input('my-graph', 'clickData')
)
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)



app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output("drag_container", "data-drag"),
    [Input("drag_container", "id")],
)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="order"),
    Output("hidden-div", "className"),
    [Input(i, 'n_clicks') for i in tags_network(FILEURL)[1]]
)


if __name__ == '__main__':
    app.run_server(debug=True)
