# ------------------------------------------------
# Create OTC graph
# ------------------------------------------------
# A graph expressing people's ratings following Bitcoin trading. Ratings can range from -10 to 10
# Nodes represent people who traded. edge represent the ranking between the outgoing vertex and the vertex the arc enters
# Vertices that were ranked between 2012-2014 are colored red and everything else is colored blue.
#The size of the vertices is affected by the total rating received - the higher the size in positive -> the higher the rating.
#The larger the size in negative -> the lower the rating
#The shape of the vertices: * Triangle for total negative ratings. * Circle for total positive ratings

from collections import defaultdict
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
import math
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import powerlaw
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
from collections import Counter
import random

matplotlib.use("TkAgg")

#-----------------------------------------------------create the graph--------------------------------------------------
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
    """
       Plots the total degree distribution of all nodes in the graph G using a log-log scale.

       Parameters:
           G (networkx.Graph): The input graph.

       Behavior:
           - Calculates the degree for each node.
           - Counts the number of nodes for each degree.
           - Plots a bar chart (log-log scale) of degree vs. node count.
           - Formats axis ticks to use plain numbers (not scientific notation).
           - Saves the plot as 'loglog_degree_dist_all_nodes.png' and displays it.
       """
    degrees = [G.degree(n) for n in G.nodes()]
    values, counts = np.unique(degrees, return_counts=True)

    plt.figure(figsize=(7, 5))
    plt.bar(values, counts, color='black',width=0.5)
    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.xlabel("Number of Nodes")
    plt.ylabel("Degree")
    plt.title("Degree Distribution")
    plt.tight_layout()
    plt.savefig("loglog_degree_dist_all_nodes.png")
    plt.show()

def get_most_connected_node(G):
    """
    Finds the node with the highest total degree in the graph G.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        node (int or str): The node ID with the highest degree.
    """
    return max(G.nodes(), key=lambda n: G.degree(n))
def analyze_node_trust(G, node_id):
    """
    Analyzes whether a node is generally trusted, distrusted, or neutral based on incoming edge weights.

    Parameters:
        G (networkx.DiGraph): Directed graph with weighted edges.
        node_id (int or str): The ID of the node to analyze.

    Behavior:
        - Sums the weights of incoming edges.
        - Prints trust status based on the sum.
    """
    if node_id in G.nodes():
        incoming_weights = [G.edges[u, v]['weight'] for u, v in G.in_edges(node_id)]
        total_rating = sum(incoming_weights)
        print(f"Node {node_id} total rating: {total_rating}")
        if total_rating > 0:
            print(f"Node {node_id} is generally trusted (positive total ratings).")
        elif total_rating < 0:
            print(f"Node {node_id} is generally distrusted (negative total ratings).")
        else:
            print(f"Node {node_id} has neutral ratings (sum is zero).")
    else:
        print(f"Node {node_id} not found in the graph.")

def plot_degree_distribution_by_color(G):
    """
       Plots the degree distribution for nodes in graph G, grouped by their 'color' attribute.
       Also analyzes the most connected node to determine its trust based on incoming edge weights.

       Parameters:
           G (networkx.Graph): The input graph where nodes may have a 'color' attribute and
                               edges have a 'weight' attribute.

       Behavior:
           - Groups node degrees by color.
           - For each color group, plots a bar chart of degree vs. node count (log-log scale).
           - Saves each plot as 'loglog_degree_nodecount_<color>.png' and displays it.
           - Finds the node with the highest degree for testing purposes
       """
    from collections import defaultdict

    degree_by_color = defaultdict(list)

    for node in G.nodes():
        color = G.nodes[node].get('color', 'blue')
        degree = G.degree(node)  # סך הדרגות
        degree_by_color[color].append(degree)

    for color, degrees in degree_by_color.items():
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1

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

    # Find and analyze the most connected node
    most_connected = get_most_connected_node(G)
    analyze_node_trust(G, most_connected)

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

#----------------------------------------------------------------------powerLaw-----------------------------------------
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

def analyze_power_law(degrees, label, color='gray'):
    """
    Fits power law to given degrees and returns the fit object.
    Also prints key statistics.
    """
    degrees = np.array(degrees)
    degrees = degrees[degrees > 0]

    if len(degrees) == 0:
        print(f"No nodes with positive degree for {label}.")
        return None

    fit = powerlaw.Fit(degrees)

    print(f"{label} - Estimated alpha: {fit.power_law.alpha:.2f}, xmin: {fit.xmin}, N: {len(degrees)}")
    return fit
def plot_combined_degree_and_powerlaw(G):
    """
    Plots combined degree distributions and power law fits for:
    - All nodes
    - Nodes grouped by color
    All in a single log-log plot for comparison.
    """
    plt.figure(figsize=(10, 7))
    # ---------- ALL NODES ----------
    degrees_all = [G.degree(n) for n in G.nodes()]
    values_all, counts_all = np.unique(degrees_all, return_counts=True)
    total_nodes_all = len(degrees_all)
    probabilities_all = counts_all / total_nodes_all
    plt.bar(values_all, probabilities_all, label='All Nodes Empirical PDF', color='grey', alpha=0.5, width=0.8)
    #plt.bar(values_all, counts_all, label='All Nodes Degree Dist.', color='black', alpha=0.5, width=0.8)

    # Power law fit for all nodes
    fit_all = analyze_power_law(degrees_all, "All Nodes")
    if fit_all:
        fit_all.plot_pdf(label='Empirical')
        fit_all.power_law.plot_pdf(color='black', linestyle='--', label='All Nodes Power Law Fit')

    # ---------- BY COLOR ----------
    # Get all existing colors in the graph
    allColors = set(attr.get('color') for _, attr in G.nodes(data=True))
    allColors.discard(None)

    for color in allColors:
        # Nodes with this color
        nodesColor = [n for n, attr in G.nodes(data=True) if attr.get('color') == color]
        degrees = np.array([G.degree(n) for n in nodesColor], dtype=int)
        degrees = degrees[degrees > 0]

        if len(degrees) == 0:
            print(f"No nodes with color {color} have positive degree.")
            continue

        # Degree distribution bar plot for this color
        total_nodes = len(degrees)
        values, counts = np.unique(degrees, return_counts=True)
        probabilities = counts / total_nodes
        plt.bar(values, probabilities, label=f'{color} Degree Dist.', alpha=0.5, width=0.5, color = color)

        # Power law fit
        fit = analyze_power_law(degrees, f"Color: {color}", color=color)
        if fit:
            fit.plot_pdf(color=color, label=f'{color} Empirical')
            fit.power_law.plot_pdf(color=color, linestyle='--', label=f'{color} Power Law Fit')

    # ---------- FORMATTING ----------
    plt.xlim(left=fit_all.xmin)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


    plt.xlabel("Degree")
    plt.ylabel("Probability P(k)")
    plt.title("Combined Degree Distributions (as PDFs) & Power Law Fits")
    plt.legend()
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------pageRank---------------------------------------
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

#---------------------------------------------------------BehavioralHomophily-------------------------------------------
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


#-----------------------------------------------------------IdentifyingCommunities--------------------------------------
def IdentifyingCommunities(G):
    """
    Detects communities in a directed graph using the Louvain method
    and analyzes color-based attributes for each community.

    Workflow:
    1. Extracts a subgraph containing only positively weighted edges.
    2. Copies relevant node attributes (e.g., color) from the original graph.
    3. Converts the subgraph too undirected to apply Louvain community detection.
    4. Computes the modularity score of the partition.
    5. Prints the distribution of node colors per community.
    6. Visualizes the communities with color-coded clusters.
    :param G:  networkx.Graph
    A directed graph where nodes may have attributes such as 'color',
    and edges contain weights representing ratings (positive or negative).

    modularity: float The modularity score of the detected partition.
    """

    def extract_positive_subgraph(G):
        """
        Filters the input graph to include only edges with positive weights.

        :param G:networkx.Graph A directed graph where nodes may have attributes such as 'color',
        and edges contain weights representing ratings (positive or negative).
        :return:A copy of the subgraph with only positively weighted edges.
        """
        return G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) > 0]).copy()

    def copy_node_attributes(original_graph, subgraph):
        """
        Copies all node attributes (such as 'color') from the original graph
        to the positive subgraph to preserve analysis metadata.

        :param original_graph: networkx.Graph
        :param subgraph: networkx.Graph
        :return:
        """
        for node, attrs in original_graph.nodes(data=True):
            if node in subgraph:
                subgraph.nodes[node].update(attrs)

    def compute_color_distribution(G, communities):
        """
        For each community, calculates the percentage of nodes
        by color attribute (e.g., 'red', 'blue').

        :param G:  G:networkx.Graph A directed graph where nodes may have attributes such as 'color',
        and edges contain weights representing ratings (positive or negative).
        :param communities: list of sets. Each set contains the nodes in a detected community
        :return: List of dicts - each dict maps color to percentage within the community.

        """
        community_percentages = []
        for i, community in enumerate(communities, 1):
            color_counts = Counter()
            total = 0

            for node in community:
                color = G.nodes[node].get('color')
                if color is not None:
                    color_counts[color] += 1
                    total += 1
            color_percentage = {}
            print(f"\nCommunity {i} - Color distribution (%):")
            if total > 0:
                for color, count in sorted(color_counts.items()):
                    percentage = (count / total) * 100
                    color_percentage[color] = percentage
                    print(f"  {color}: {percentage:.2f}%")
            else:
                print("  No color data available.")
            community_percentages.append(color_percentage)
        return community_percentages

    def draw_communities_colored(G, communities, title="Community Graph"):
        """
        Visualizes the graph with nodes colored by community assignment.
        G : G:networkx. Graph A directed graph where nodes may have attributes such as 'color',
        and edges contain weights representing ratings (positive or negative).
        communities : list of sets. Each set contains the nodes in a detected community.
        title : str
            Title of the plot.

        :param G:
        :param communities:
        :param title:
        :return:
        """
        color_distributions = compute_color_distribution(G_undirected, communities)
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 8))

        for i, community in enumerate(communities):
            label = f"Community {i + 1}"

            # If color distribution provided, add it to label
            if color_distributions and i < len(color_distributions):
                parts = [f"{color}:{pct:.1f}%" for color, pct in sorted(color_distributions[i].items())]
                label += " (" + ", ".join(parts) + ")"

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=list(community),
                node_color=f"C{i}",
                label=label,
                node_size=100,
                alpha=0.8
            )

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title(title)
        plt.axis('off')
        plt.legend()
        plt.show()

    G_pos = extract_positive_subgraph(G)
    copy_node_attributes(G, G_pos)
    G_undirected = G_pos.to_undirected()

    communities = louvain_communities(G_undirected, weight='weight')
    mod = modularity(G_undirected, communities, weight='weight')

    print(f"\n✅ Modularity: {mod:.3f}")
    print(f"✅ Number of communities found: {len(communities)}")
    compute_color_distribution(G_undirected, communities)
    draw_communities_colored(G_undirected, communities)

#-----------------------------------------Spread IC --------------------------------------------------------------------

def CalculateTheProb(value):
    """
       Calculates the activation probability of an edge in an Information Cascade (IC) model
       based on its weight. This function is designed to simulate the likelihood of information
       propagation in social networks, where edges represent connections between users,
       and their weights influence the probability of information spreading.

       The formula used is:
       probability = max(0.005, min(0.1 + (value - 1) * 0.08, 0.9))

       Parameters:
       - value (float or int): The weight of the edge, representing the strength of the connection.

       Returns:
       - probability (float): Activation probability, bounded between 0.005 and 0.9,
         rounded to two decimal places.

       Explanation of the formula:
       1. Base Probability (0.1):
          - When the edge weight is 1, the base activation probability is set to 0.1.
          - This ensures that even weak connections have a non-negligible chance of spreading information.

       2. Weight Scaling Factor (0.08):
          - The value 0.08 is chosen to balance the sensitivity of the model.
          - It ensures a gradual increase in activation probability as the edge weight increases,
            preventing extreme jumps in probability for small changes in weight.
          - This factor is empirically derived to match real-world observations in social networks,
            where stronger connections (higher weights) moderately increase the likelihood of information spread.

       3. Bounds (0.005 and 0.9):
          - The lower bound (0.005) prevents the probability from becoming negligible for very weak edges.
          - The upper bound (0.9) ensures that even the strongest edges do not guarantee deterministic spread,
            reflecting the inherent uncertainty in real-world information cascades.

       Example:
       - For value = 1: probability = 0.1
       - For value = 5: probability = 0.1 + (5-1)*0.08 = 0.42
       - For value = 10: probability = 0.9 (capped at the upper bound)
       """
    probability = max(0.005, min(0.1 + (value - 1) * 0.08, 0.9))
    return int(probability * 100) / 100

def AddProbToGraph():
    """
       Assigns a 'prob' attribute to all edges in the graph based on their weights.
       Uses the CalculateTheProb function for each edge.

       Parameters:
       - Uses the global 'graph' object

       Returns:
       - None (modifies the graph in place)
       """
    for u in graph.nodes():
        for v, _, data in graph.in_edges(u, data=True):
            data['prob'] = CalculateTheProb(data['weight'])


def AvarageGroups(groups):
    """
        Takes multiple matrices of activation counts and computes the average
        across all runs and all groups for each iteration step.

        Parameters:
        - groups (list of np.ndarray): Each matrix has shape (num_simulations, num_steps)

        Returns:
        - np.ndarray: 1D array containing the average number of newly activated nodes per iteration step
        """
    max_len = max(g.shape[1] for g in groups)
    padded = []

    for g in groups:
        mat = np.zeros((g.shape[0], max_len))
        mat[:, :g.shape[1]] = g
        padded.append(mat)

    all_runs = np.vstack(padded)

    return all_runs.mean(axis=0)

def PlotAveragePerGroup(groups, group_names):
    """
    Plots a separate graph for each group showing the average number
    of newly activated nodes per iteration step across all runs.

    Parameters:
    - groups (list of np.ndarray): Each element is the matrix returned from CalculateIter
    - group_names (list of str): Names of the groups for labeling the plots
    """
    for matrix, name in zip(groups, group_names):
        # ממוצע לפי איטרציה (עמודה) על כל הריצות
        avg_per_iteration = matrix.mean(axis=0)

        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(avg_per_iteration)+1), avg_per_iteration, marker='o')
        plt.title(f"Average Spread per Iteration - {name}")
        plt.xlabel("Iteration Step")
        plt.ylabel("Average Newly Activated Nodes")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{name}_average.png", dpi=300)
        plt.show()


def CreateGroupsColors(color):
    """
       Creates two groups of nodes based on color: the last 30 nodes by out-degree
       and the most connected node.

       Parameters:
       - color (str): 'red' or 'blue'

       Returns:
       - group_last (list): Last 30 nodes sorted by out-degree (least connected)
       - group_most_connected (node): Node with the highest out-degree
       """

    nodes = [(n, graph.out_degree(n)) for n in graph.nodes() if graph.nodes[n].get("color") == color]
    sorted_nodes = [n for n, _ in sorted(nodes, key=lambda x: x[1], reverse=True)]

    group_last = sorted_nodes[-30:]
    group_most_connected = sorted_nodes[0]

    return group_last, group_most_connected

def PrintGraph(final_avg):
    """
       Plots a line graph of the average number of newly activated nodes per iteration step.

       Parameters:
       - final_avg (np.ndarray): 1D array of average newly activated nodes per step

       Returns:
       - None (displays a matplotlib plot)
       """
    plt.figure(figsize=(8,5))
    plt.plot(range(len(final_avg)), final_avg, marker='o')
    plt.title("Average Spread per Iteration Step")
    plt.xlabel("Iteration Step")
    plt.ylabel("Average Newly Activated Nodes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("Average.png", dpi=300)
    plt.show()

def CalculateIter(seed_node):
    """
        Runs multiple simulations (70 by default) of the Independent Cascade (IC) model
        starting from the given seed node(s), recording how many nodes are activated
        per iteration.

        Parameters:
        - seed_node (list or set): Nodes to start the spread from

        Returns:
        - matrix (np.ndarray): Shape (70, max_steps), number of newly activated nodes per iteration
        - avg_red (float): Average number of red nodes activated per simulation
        - avg_blue (float): Average number of blue nodes activated per simulation
        """
    all_iter_counts = []
    all_red_counts = []
    all_blue_counts = []

    for _ in range(70):
        active = set(seed_node)
        newly_active = set(seed_node)

        iter_counts = []
        red_counts = []
        blue_counts = []

        while newly_active:
            next_new = set()
            for node in newly_active:
                for _, neighbor, data in graph.out_edges(node, data=True):
                    if neighbor not in active:
                        if random.random() < data['prob']:
                            next_new.add(neighbor)
            active.update(next_new)
            iter_counts.append(len(next_new))
            red_counts.append(len(next_new & red_nodes))
            blue_counts.append(len(next_new & blue_nodes))
            newly_active = next_new

        all_iter_counts.append(iter_counts)
        all_red_counts.append(sum(red_counts))
        all_blue_counts.append(sum(blue_counts))

    max_run_length = max(len(run) for run in all_iter_counts)
    matrix = np.zeros((70, max_run_length), dtype=int)

    for row_number, run in enumerate(all_iter_counts):
        run_length = len(run)
        matrix[row_number, :run_length] = run

    return matrix, all_red_counts,all_blue_counts

def PrintGraphDual(red_from_blue, blue_from_red):
    """
    Plots a line graph showing:
    - Red nodes activated from blue seeds
    - Blue nodes activated from red seeds

    Parameters:
    - red_from_blue (np.ndarray): Average newly activated red nodes per step from blue seeds
    - blue_from_red (np.ndarray): Average newly activated blue nodes per step from red seeds
    """
    plt.figure(figsize=(8,5))
    plt.plot(range(len(red_from_blue)), red_from_blue, marker='o', color='red', label='Red from Blue seeds')
    plt.plot(range(len(blue_from_red)), blue_from_red, marker='o', color='blue', label='Blue from Red seeds')
    plt.title("Cross-color Spread per Iteration Step")
    plt.xlabel("Iteration Step")
    plt.ylabel("Average Newly Activated Nodes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig("cross_color_spread.png", dpi=300)
    plt.show()


def SpreadIC():
    """
       Main function to run the Independent Cascade model on predefined groups of nodes,
       calculate average activations, plot the spread over iterations, and print
       the average percentages of red and blue nodes activated.

       Parameters:
       - None (uses the global 'graph', 'red_nodes', and 'blue_nodes')

       Returns:
       - None (prints percentages and shows the graph)
       """

    AddProbToGraph()

    GroupRedLast30,GroupRedMostConnected = CreateGroupsColors("red")
    GroupBlueLast30,GroupBlueMostConnected = CreateGroupsColors("blue")

    group_names = ["Red Last 30", "Red Most Connected", "Blue Last 30", "Blue Most Connected"]

    AllGroups = []
    red_totals = []
    blue_totals = []

    for group in [GroupRedLast30, [GroupRedMostConnected], GroupBlueLast30, [GroupBlueMostConnected]]:
        mat, red, blue = CalculateIter(group)
        AllGroups.append(mat)
        red_totals.append(red)
        blue_totals.append(blue)

    PlotAveragePerGroup(AllGroups, group_names)

    all_red_matrix = np.vstack(red_totals)  # כל מטריצה של קבוצות מוערמת למטריצה אחת
    all_blue_matrix = np.vstack(blue_totals)

    # ממוצע לפי עמודה (שלב)
    red_activated_avg_per_step = np.mean(all_red_matrix, axis=0)
    blue_activated_avg_per_step = np.mean(all_blue_matrix, axis=0)

    # אחוזים לפי שלב
    red_percent_per_step = red_activated_avg_per_step / len(red_nodes) * 100
    blue_percent_per_step = blue_activated_avg_per_step / len(blue_nodes) * 100

    BigGroups = AvarageGroups(AllGroups)
    PrintGraph(BigGroups)

    print(f"Red activated: {np.mean(red_percent_per_step):.2f}%")
    print(f"Blue activated: {np.mean(blue_percent_per_step):.2f}%")
#----------------------------------------------------------- Start------------------------------------------------------

file_path = "largest_scc_edges.csv"
chunksize = 100000  # Number of lines in each chunk

graph = nx.DiGraph()  # Creating an empty directed graph

# Reading the file in chunks and gradually adding the information to the graph
with open(file_path, 'rt') as f:
    for chunk in pd.read_csv(f, names=["source", "target", "weight", "time"], skiprows=1, chunksize=chunksize):
        # Convert time column to datetime column - format "%d/%m/%Y"
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

#-----------------------------------calls for the functions-------------------------------------------------------------
settingOfSCCgraph(graph)
#BehavioralHomophily(graph)
#printInformSourceGraph()
#colors = [node_colors[node] for node in G_my.nodes()]
#plot_total_degree_distribution(graph)
#plot_degree_distribution_by_color(graph)
#powerLaw(graph)
#printInformOfBiggestSCCGraph(graph)
#pageRank(graph)
#plot_combined_degree_and_powerlaw(graph)
#IdentifyingCommunities(graph)
# colors = [node_colors[node] for node in G_my.nodes()]
# nx.draw(G_my, with_labels=True, node_color=colors, edge_color='gray')
red_nodes = {n for n in graph.nodes() if graph.nodes[n].get("color") == "red"}
blue_nodes = set(graph.nodes()) - red_nodes
SpreadIC()



#------------------another version of SpreadIC---------------------------------

def SpreadICVer02():
    """
    #        Main function to run the Independent Cascade model on predefined groups of nodes,
    #        calculate average activations, plot the spread over iterations, and print
    #        the average percentages of red and blue nodes activated.
    #
    #        Parameters:
    #        - None (uses the global 'graph', 'red_nodes', and 'blue_nodes')
    #
    #        Returns:
    #        - None (prints percentages and shows the graph)
    #        """
    AddProbToGraph()

    GroupRedLast30, GroupRedMostConnected = CreateGroupsColors("red")
    GroupBlueLast30, GroupBlueMostConnected = CreateGroupsColors("blue")

    red_activated_from_blue = []
    blue_activated_from_red = []

    # ריצות קבוצות אדומות
    for group in [GroupRedLast30, [GroupRedMostConnected]]:
        _, red, blue = CalculateIter(group)
        # כמה כחולים נדלקו מהריצה של אדומים
        blue_activated_from_red.append(blue)

    # ריצות קבוצות כחולות
    for group in [GroupBlueLast30, [GroupBlueMostConnected]]:
        _, red, blue = CalculateIter(group)
        # כמה אדומים נדלקו מהריצה של כחולים
        red_activated_from_blue.append(red)

    # ממיר לכל מטריצה
    red_from_blue_matrix = np.vstack(red_activated_from_blue)
    blue_from_red_matrix = np.vstack(blue_activated_from_red)

    # ממוצע לפי שלב
    red_from_blue_avg_per_step = np.mean(red_from_blue_matrix, axis=0)
    blue_from_red_avg_per_step = np.mean(blue_from_red_matrix, axis=0)

    # גרף
    PrintGraphDual(red_from_blue_avg_per_step, blue_from_red_avg_per_step)

    # הדפסה של אחוז כולל
    print(f"Red activated from blue seeds (overall avg): {np.mean(red_from_blue_avg_per_step / len(red_nodes) * 100):.2f}%")
    print(f"Blue activated from red seeds (overall avg): {np.mean(blue_from_red_avg_per_step / len(blue_nodes) * 100):.2f}%")
