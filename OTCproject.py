# ------------------------------------------------
# Create OTC graph
# ------------------------------------------------
# A graph expressing people's ratings following Bitcoin trading. Ratings can range from -10 to 10
# Nodes represent people who traded. edge represent the ranking between the outgoing vertex and the vertex the arc enters
# Vertices that were ranked between 2012-2014 are colored red and everything else is colored blue.
#The size of the vertices is affected by the total rating received - the higher the size in positive -> the higher the rating.
#The larger the size in negative -> the lower the rating
#The shape of the vertices: * Triangle for total negative ratings. * Circle for total positive ratings
from collections import defaultdict,Counter
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def settingOfSCCgraph (G):
    # Setting a color for each vertex
    node_colors = {}
    for node in G.nodes():
        incoming_years = [G.edges[u, v]['year'] for u, v in G.in_edges(node)]
        if incoming_years:  # If there are incoming arcs
            last_year = max(incoming_years)
            #print(last_year)
            if 2014 <= last_year <= 2016:
                node_colors[node] = 'red'
            else:
                node_colors[node] = 'blue'
        else:
            node_colors[node] = 'blue'  # Default for those who do not have an incoming arc

        G.nodes[node]['color'] = node_colors[node]


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

    #Saving the node size as its attribute
    for node, size in normalized_sizes.items():
        G.nodes[node]['size'] = size

    #Saving the shape of the node as its attribute
    for node in G.nodes():
        if node_sentiment[node] >= 0:
            G.nodes[node]['shape'] = 'o'
        else:
            G.nodes[node]['shape'] = '^'

    #Graph drawing
    def drawingGraph(G):
        pos = nx.spring_layout(G, seed=42, k=0.3)
        plt.figure(figsize=(14, 10))

        positiveNodes = [node for node in G.nodes() if G.nodes[node]['shape'] == 'o']
        negativeNodes = [node for node in G.nodes() if G.nodes[node]['shape'] == '^']

        # Drawing positive nodes (circles)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=positiveNodes,
            node_size=[normalized_sizes[n] for n in positiveNodes],
            node_color=[node_colors[n] for n in positiveNodes],
            node_shape='o',
            edgecolors='black',
            alpha=0.9
        )

        # Drawing negative nodes (triangles)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=negativeNodes,
            node_size=[normalized_sizes[n] for n in negativeNodes],
            node_color=[node_colors[n] for n in negativeNodes],
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
    #drawingGraph(G)

def plot_total_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    values, counts = np.unique(degrees, return_counts=True)

    plt.figure(figsize=(7, 5))
    plt.bar(values, counts, color='black',width=0.5)
    plt.xscale('log')
    plt.yscale('log')

    # פורמט הטיקים להצגת הערך האמיתי ולא בצורה מדעית
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Number of Nodes")
    plt.ylabel("Degree")
    plt.title("Degree Distributionn")
    plt.tight_layout()
    plt.savefig("loglog_degree_dist_all_nodes.png")
    plt.show()


def plot_degree_distribution_by_color(G):
    degree_by_color = defaultdict(list)

    # מחשבים את הדרגה הכוללת לכל קודקוד ומחלקים לפי צבע
    for node in G.nodes():
        color = G.nodes[node].get('color', 'blue')
        degree = G.degree(node)  # סך הדרגות
        degree_by_color[color].append(degree)

    for color, degrees in degree_by_color.items():
        # סופרים כמה קודקודים יש לכל דרגה
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1

        # מפרידים ל־X ו־Y: כמה קודקודים יש עם דרגה מסוימת
        sorted_degrees = sorted(degree_counts.items())
        x = [count for degree, count in sorted_degrees]
        y = [degree for degree, count in sorted_degrees]

        plt.figure(figsize=(8, 6))
        plt.bar(x, y, color=color, edgecolor='black', width=0.5)

        plt.xscale('log')
        plt.yscale('log')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        plt.xlabel("Number of Nodes")
        plt.ylabel("Degree Value")
        plt.title("Degree Distribution")
        plt.tight_layout()
        plt.savefig(f"loglog_degree_nodecount_{color}.png")
        plt.show()
    #רצינו לבדוק את הקודקוד המפורסם זה הכי מקושר לכולם האם הוא מדורג בתור אמין ולכן הוא כל כך מקושר
    node_id = 35  # הקודקוד שרוצים לבדוק

    if node_id in graph.nodes():
        incoming_weights = [graph.edges[u, v]['weight'] for u, v in graph.in_edges(node_id)]
        total_rating = sum(incoming_weights)
        print(f"Node {node_id} total rating: {total_rating}")
        if total_rating > 0:
            print("Node 35 is generally trusted (received positive ratings).")
        elif total_rating < 0:
            print("Node 35 is generally distrusted (received negative ratings).")
        else:
            print("Node 35 has neutral ratings (sum is zero).")
    else:
        print(f"Node {node_id} not found in the graph.")
def printInformSourceGraph():
    """
    A function that prints the following data of the source graph:
    Number of vertices
    Number of edges
    Number of connecting elements
    Number of edges of the largest connecting element
    :return: None
    """
    file_path = "BitCoin.csv"
    chunksize = 100000  # Number of lines in each chunk

    G = nx.DiGraph()  # Creating an empty directed graph

    # Reading the file in chunks and gradually adding the information to the graph
    with open(file_path, 'rt') as f:
        for chunk in pd.read_csv(f, names=["source", "target", "weight", "time"], skiprows=1, chunksize=chunksize):
            chunk["time"] = pd.to_datetime(chunk["time"], format="%d/%m/%Y")
            for _, row in chunk.iterrows():
                G.add_edge(row["source"], row["target"], weight=row["weight"], time=row["time"].timestamp())

    # Analyze strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    print(f"\n🔍 Found {len(sccs)} strongly connected components.")

    # Sort by size
    sccs_sorted = sorted(sccs, key=len, reverse=True)

    for i, component in enumerate(sccs_sorted[:10], start=1):  # Show top 10 components
        print(f"Component #{i} contains {len(component)} nodes.")

    largest_scc = sccs_sorted[0]
    print(f"\n⭐ The largest component contains {len(largest_scc)} nodes.")

    # print number of edges in the largest component
    largest_scc_subgraph = G.subgraph(largest_scc)
    print(f"   It has {largest_scc_subgraph.number_of_edges()} edges.")
    print(f"Number of nodes in the graph: {G.number_of_nodes()}")
    print(f"Number of edges in the graph: {G.number_of_edges()}")

    # Graph drawing
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=500,
        font_size=8,
        arrows=True,
        arrowsize=15
    )
    plt.title("Directed Graph Visualization")
    plt.show()


def printInformOfBiggestSCCGraph(G):
    """
    A function that prints the data on the subgraph - the largest SCC
    :param G: A graph constructed from networkX
    :return:
    """

    #Printing the number of vertices of each color
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    color_counts = Counter(colors)
    for color, count in color_counts.items():
        print(f"{color}: {count} nodes")

    positiveCount = sum(1 for node in G.nodes() if G.nodes[node].get('shape') == 'o')
    negativeCount = sum(1 for node in G.nodes() if G.nodes[node].get('shape') == '^')
    print(f"Number of positively rated nodes (circles): {positiveCount}")
    print(f"Number of negatively rated nodes (triangles): {negativeCount}")

def BehavioralHomophily(G):
    """
    This function analyzes behavioral homophily in a directed user rating graph.
    Specifically, it tests whether users who tend to give positive ratings
    (e.g., "Positive raters") are more likely to receive positive ratings
    from users with similar rating behavior.

    Steps:
    1. Classify users by their rating pattern:
    - For each user, compute the average rating they give (i.e., average weight of outgoing edges).
    - Label as:
    "Positive rater" if average rating > 2,
    "Negative rater" if average rating < -2,
    "Neutral" otherwise.

    2. For each edge, check whether the source and target belong to the same group.
    - Count edges where both users are in the same group (Positive/Negative/Neutral).
    - Compute behavioral homophily index:
    H = (number of edges between users in the same group) / (total number of edges)

    :param G: A graph constructed from networkX
    :return: None
    """


    # --- Assigning users to groups based on the average ratings they give --
    DividingUsersToGroups = {}
    for node in G.nodes:
        outEdges = G.out_edges(node, data=True)
        if not outEdges:
            DividingUsersToGroups[node] = 'neutral'
            continue

        ratings = [data['weight'] for (_, _, data) in outEdges]
        avg = sum(ratings) / len(ratings)
        if avg > 2:
            DividingUsersToGroups[node] = 'positive'
        elif avg < -2:
            DividingUsersToGroups[node] = 'negative'
        else:
            DividingUsersToGroups[node] = 'neutral'

    # --- Cross-team edges statistics  ---
    edgeStats = defaultdict(lambda: defaultdict(int)) # Nested Dictionary Data Structure
    for u, v in G.edges:
        if u in DividingUsersToGroups and v in DividingUsersToGroups:
            edgeStats[DividingUsersToGroups[u]][DividingUsersToGroups[v]] += 1
    print(edgeStats)

    # --- Calculating homophily rates for the entire group ---
    HomophilyPerGroup = []
    for group in ['positive', 'neutral', 'negative']:
        groupOutEdges = edgeStats[group] #The groups that have edges from the current group
        totalEdgesFromGroup = sum(groupOutEdges.values()) # Total outgoing edges for each group
        if totalEdgesFromGroup == 0:
            HomophilyPerGroup.append((group, 0, 0, 0))
            continue
        sameGroup = groupOutEdges.get(group, 0)
        sameRatio = sameGroup / totalEdgesFromGroup
        HomophilyPerGroup.append((group, totalEdgesFromGroup, sameGroup, sameRatio))


    totalEdgesInGraph = G.number_of_edges()
    sameGroupTotal = sum(edgeStats[g][g] for g in edgeStats)
    overallHomophily = sameGroupTotal / totalEdgesInGraph if totalEdgesInGraph > 0 else 0

    groups = ['positive', 'neutral', 'negative']

    # --- Describing out edges to the different groups in a bar graph ---
    # Data: From HomophilyPerGroup calculates the ratio within and outside the group
    sameRatios = []
    differentRatios = []

    for group, total, same, sameRatio in HomophilyPerGroup:
        sameRatios.append(sameRatio)
        differentRatios.append(1 - sameRatio)

    # Column location
    x = np.arange(len(groups))
    width = 0.35  # width of each column

    # drawing bar graph
    fig, ax = plt.subplots()
    sameRatiosBars = ax.bar(x - width / 2, sameRatios, width, label='same group', color='skyblue')
    OtherGroupsBars = ax.bar(x + width / 2, differentRatios, width, label='Other groups', color='lightcoral')

    # designing of bar graph
    ax.set_ylabel('Rating rate')
    ax.set_title('Rating rate for each group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    # Adding values to the columns
    for bar in sameRatiosBars + OtherGroupsBars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset to text height
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # --- printing results ---
    results_text = "\n📊 Ranking rates within and between groups:\n"
    for group, total, same, ratio in HomophilyPerGroup:
        if total == 0:
            results_text += f"{group}: No outgoing edges\n"
        else:
            results_text += (
                f"• {group}:\n"
                f"  - total out edges: {total}\n"
                f"  - To the same group: {same} ({ratio:.2%})\n"
                f"  - For other groups : {total - same} ({1 - ratio:.2%})\n\n"
            )

    results_text += f"📈 General homophily index (all edges within the same group): {overallHomophily:.2%}"
    print (results_text)


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

# settingOfSCCgraph(graph)
# BehavioralHomophily(graph)
#printInformSourceGraph()

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

settingOfSCCgraph(graph)
plot_total_degree_distribution(graph)
plot_degree_distribution_by_color(graph)