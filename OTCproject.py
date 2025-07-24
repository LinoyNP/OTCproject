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
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
matplotlib.use("TkAgg")

def settingOfSCCgraph (G):
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

import matplotlib.ticker as ticker

def plot_total_degree_distribution(G):
    degrees = [G.degree(n) for n in G.nodes()]
    values, counts = np.unique(degrees, return_counts=True)

    plt.figure(figsize=(7, 5))
    plt.bar(values, counts, color='gray', edgecolor='black', width=0.8)
    plt.xscale('log')
    plt.yscale('log')

    # פורמט הטיקים להצגת הערך האמיתי ולא בצורה מדעית
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Degree (log scale)")
    plt.ylabel("Number of Nodes (log scale)")
    plt.title("Total Degree Distribution (Log-Log Scale)")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("loglog_degree_dist_all_nodes.png")
    plt.show()

def plot_degree_distribution_by_color(G):
    from collections import defaultdict

    degree_by_color = defaultdict(list)

    for node in G.nodes():
        color = G.nodes[node].get('color', 'blue')  # Default is blue
        degree = G.degree(node)
        degree_by_color[color].append(degree)

    for color, degrees in degree_by_color.items():
        values, counts = np.unique(degrees, return_counts=True)

        plt.figure(figsize=(7, 5))
        plt.bar(values, counts, color=color, edgecolor='black', width=0.8)
        plt.xscale('log')
        plt.yscale('log')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        plt.xlabel("Degree (log scale)")
        plt.ylabel("Number of Nodes (log scale)")
        plt.title(f"Degree Distribution (Log-Log) - {color.capitalize()} Nodes")
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"loglog_degree_dist_{color}.png")
        plt.show()


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
    print(f"average shortest path = {nx.average_shortest_path_length(G)}")
def powerLaw(G):
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    degrees = degrees[degrees > 0]
    fit = powerlaw.Fit(degrees)

    # --- power law of all degrees ---
    #fit = powerlaw.Fit(degrees[degrees > 0])
    # Graphing: Rank distribution vs. fit to a power law
    plt.figure()
    fit.plot_pdf(label='Empirical')
    fit.power_law.plot_pdf(color='r', linestyle='--', label='Power law fit')
    plt.xlabel("Degree")
    plt.ylabel("P(k)")
    plt.legend()
    plt.title("Power-law Degree Distribution OTC")
    plt.show()
    # print power-law exponent (beta)
    print(f"Estimated power-law exponent (beta): {fit.power_law.alpha:.2f}")
    print(f"the xmin is: {fit.xmin}")

    def analyzeColor(G, colorName):
        """
        Analyzes the degree distribution of nodes with a given color in the graph,
        fits a power law model, and prints key statistics.
        :param G: networkx.Graph The input graph. Each node is expected to have 'color' and 'shape' attributes
        :param colorName: str that represents the color of node
        :return: powerlaw.Fit or None
                Returns a powerlaw.Fit object containing the power law fit results
                for nodes with the specified color.
                If no nodes with positive degree exist for this color, returns None
        """

        # Nodes with the desired color
        nodesColor = [n for n, attr in G.nodes(data=True) if attr.get('color') == colorName]

        # The degrees of the nodes in this color
        degrees = np.array([G.degree(n) for n in nodesColor], dtype=int)
        degrees = degrees[degrees > 0]

        if len(degrees) == 0:
            print(f"No nodes with color {colorName} have positive degree.")
            return None

        fit = powerlaw.Fit(degrees)

        print(f"Color: {colorName}")
        print(f"  Estimated alpha: {fit.power_law.alpha:.2f}")
        print(f"  xmin: {fit.xmin}")
        print(f"  Number of nodes: {len(degrees)}")

        return fit

    #Representation of the possession law for each color
    allColors = set(attr.get('color') for _, attr in G.nodes(data=True))
    allColors.discard(None)

    # --- Analysis for each color ---
    fitsByColor = {}
    for color in allColors:
        fit = analyzeColor(G, color)
        if fit is not None:
            fitsByColor[color] = fit

    # --- Drawing all the graphs (power law for each color) side by side ---
    numColors = len(fitsByColor)
    plt.figure(figsize=(5*numColors,5))  # Size according to the number of colors

    for i, (color, fit) in enumerate(fitsByColor.items(), start=1):
        plt.subplot(1, numColors, i)
        fit.plot_pdf(label='Empirical')
        fit.power_law.plot_pdf(linestyle='--', label='Power law fit')
        plt.xlabel("Degree")
        plt.ylabel("P(k)")
        plt.title(f"Color: {color}")
        plt.legend()

    plt.tight_layout()
    plt.show()

def pageRank(G):
    # ===Build a positive-weights subgraph from SCC_G ===
    Gpositive = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0:
            Gpositive.add_edge(u, v, weight=data['weight'])
    pagerank = nx.pagerank(Gpositive, weight='weight')  # PageRank
    top10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 nodes by pagerank:")
    for node, val in top10:
        print(f"{node}: {val:.4f}")
    reciprocalPositivePairs = []
    for i in top10:
        for j in top10:
            if i != j:
                if Gpositive.has_edge(i, j) and G[i][j].get("weight", 0) > 0:
                    print("yes")
                    if Gpositive.has_edge(j, i) and G[j][i].get("weight", 0) > 0:
                        reciprocalPositivePairs.append((i, j))

    print("Top user pairs with mutually positive ratings")
    if len(reciprocalPositivePairs) == 0:
        print("No pairs of top users with mutually positive ratings were found.")
    for pair in reciprocalPositivePairs:
        print(f"{pair[0]} ↔ {pair[1]}")

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

settingOfSCCgraph(graph)
# plot_total_degree_distribution(graph)
# plot_degree_distribution_by_color(graph)
#powerLaw(graph)
#printInformOfBiggestSCCGraph(graph)
pageRank(graph)

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

