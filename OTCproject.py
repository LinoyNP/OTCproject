# ------------------------------------------------
# Create OTC graph
# ------------------------------------------------
# A graph expressing people's ratings following Bitcoin trading. Ratings can range from -10 to 10
# Nodes represent people who traded. edge represent the ranking between the outgoing vertex and the vertex the arc enters
# Vertices that were ranked between 2012-2014 are colored red and everything else is colored blue.
#The size of the vertices is affected by the total rating received - the higher the size in positive -> the higher the rating.
#The larger the size in negative -> the lower the rating
#The shape of the vertices: * Triangle for total negative ratings. * Circle for total positive ratings

import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def DrawingGraph (G):
    # Setting a color for each vertex
    node_colors = {}
    for node in G.nodes():
        incoming_years = [G.edges[u, v]['year'] for u, v in G.in_edges(node)]
        if incoming_years:  # If there are incoming arcs
            last_year = max(incoming_years)
            print(last_year)
            if 2014 <= last_year <= 2016:
                node_colors[node] = 'red'
            else:
                node_colors[node] = 'blue'
        else:
            node_colors[node] = 'blue'  # Default for those who do not have an incoming arc


    # Calculating the sum of ratings the node received
    node_sentiment = {}
    for node in G.nodes():
        weights = [G.edges[u, v]['weight'] for u, v in G.in_edges(node)]
        total = sum(weights) if weights else 0
        node_sentiment[node] = total

    # Normalization of sizes to graphically adjust the vertex sizes according to the number of ratings it received
    sizes_raw = {node: abs(score) for node, score in node_sentiment.items()}
    max_size = max(sizes_raw.values()) if sizes_raw else 1
    normalized_sizes = {node: 300 + 1200 * (size / max_size) for node, size in sizes_raw.items()}

    pos = nx.spring_layout(G, seed=42, k=0.3)
    plt.figure(figsize=(14, 10))

    positive_nodes = [node for node in G.nodes() if node_sentiment[node] >= 0]
    negative_nodes = [node for node in G.nodes() if node_sentiment[node] < 0]

    # Drawing positive nodes (circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=positive_nodes,
        node_size=[normalized_sizes[n] for n in positive_nodes],
        node_color=[node_colors[n] for n in positive_nodes],
        node_shape='o',
        edgecolors='black',
        alpha=0.9
    )

    # Drawing negative nodes (triangles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=negative_nodes,
        node_size=[normalized_sizes[n] for n in negative_nodes],
        node_color=[node_colors[n] for n in negative_nodes],
        node_shape='^',
        edgecolors='black',
        alpha=0.9
    )

    # edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

    # labels
    # nx.draw_networkx_labels(G_my, pos, font_size=9)

    plt.title("Trust Graph with Sentiment-based Shapes and Year-based Colors")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

file_path = "largest_scc_edges.csv"
chunksize = 100000  # Number of lines in each chunk

graph = nx.DiGraph()  # Creating an empty directed graph

# Reading the file in chunks and gradually adding the information to the graph
with open(file_path, 'rt') as f:
    for chunk in pd.read_csv(f, names=["source", "target", "weight", "time"], skiprows=1, chunksize=chunksize):
        #Convert time column to datetime column - format "%d/%m/%Y"
        chunk["time"] = pd.to_datetime(chunk["time"], format="%d/%m/%Y", errors="coerce")
        chunk = chunk.dropna(subset=["time"])  # Removing rows with invalid dates
        chunk["year"] = chunk["time"].dt.year

        for _, row in chunk.iterrows():
            graph.add_edge(
                row["source"],
                row["target"],
                weight=row["weight"],
                year=row["year"]  # שמור את השנה כ-attribute
            )


DrawingGraph(graph)



# colors = [node_colors[node] for node in G_my.nodes()]
# nx.draw(G_my, with_labels=True, node_color=colors, edge_color='gray')

# pos = nx.spring_layout(G_my)
# # nx.draw(
# #     G_my,
# #     pos,
# #     with_labels=True,
# #     node_color=colors,
# #     edge_color='gray',
# #     node_size=400,
# #     font_size=10
# # )
# plt.show()