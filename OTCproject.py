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
import matplotlib.pyplot as plt
import powerlaw
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
from collections import Counter
import random

matplotlib.use("TkAgg")


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

def IdentifyingCommunities(G):
    """
    Detects communities in a directed graph using the Louvain method
    and analyzes color-based attributes for each community.

    Workflow:
    1. Extracts a subgraph containing only positively weighted edges.
    2. Copies relevant node attributes (e.g., color) from the original graph.
    3. Converts the subgraph to undirected to apply Louvain community detection.
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

#
# def sigmoid(x, alpha):
#     return 1 / (1 + math.exp(-alpha * x))
#
# alpha = 0.01
# for u in graph.nodes():
#     for v, _, data in graph.in_edges(u, data=True):
#         data['prob'] = sigmoid(data['weight'], alpha)
#
#
# def run_ic(graph, seed_node):
#
#     next_new = 1
#     active = set([seed_node])
#     newly_active = set([seed_node])
#     iter_counts = []
#
#     red_nodes = {n for n in graph.nodes() if graph.nodes[n].get("color") == "red"}
#     blue_nodes = set(graph.nodes()) - red_nodes
#
#     red_counts = []
#     blue_counts = []
#
#     while newly_active and len(active) < len(graph.nodes()):
#         next_new = set()
#         for node in newly_active:
#             for _, neighbor, data in graph.out_edges(node, data=True):
#                 if neighbor not in active:
#                     if random.random() < data['prob']:  # עכשיו המשקל הוא ההסתברות
#                         next_new.add(neighbor)
#         active.update(next_new)
#         newly_active = next_new
#         red_counts.append(len(next_new & red_nodes))
#         blue_counts.append(len(next_new & blue_nodes))
#         iter_counts.append(len(next_new))
#
#     return iter_counts, red_counts, blue_counts
#
# def influence(graph, seed_node, n_sim=1000):
#     max_iters = 0
#     all_iter_counts = []
#     all_red_counts = []
#     all_blue_counts = []
#
#     for _ in range(n_sim):
#         iter_counts, red_counts, blue_counts = run_ic(graph, seed_node)
#         max_iters = max(max_iters, len(iter_counts))
#         all_iter_counts.append(iter_counts)
#         all_red_counts.append(red_counts)
#         all_blue_counts.append(blue_counts)
#
#     # ממוצע לכל איטרציה (לפי האורך המקסימלי)
#     avg_total = np.zeros(max_iters)
#     avg_red = np.zeros(max_iters)
#     avg_blue = np.zeros(max_iters)
#
#     for sim in range(n_sim):
#         for i, val in enumerate(all_iter_counts[sim]):
#             avg_total[i] += val / n_sim
#         for i, val in enumerate(all_red_counts[sim]):
#             avg_red[i] += val / n_sim
#         for i, val in enumerate(all_blue_counts[sim]):
#             avg_blue[i] += val / n_sim
#
#     return avg_total, avg_red, avg_blue
#
# # שימוש
# red_nodes = [n for n in graph.nodes() if graph.nodes[n].get("color") == "red"]
# sorted_red_nodes = sorted(red_nodes, key=lambda n: graph.in_degree(n) + graph.out_degree(n), reverse=True)
# seed_node = sorted_red_nodes[0]
# avg_total, avg_red, avg_blue = influence(graph, seed_node, n_sim=1000)
#
#
# iterations = range(1, len(avg_total)+1)
#
# plt.figure(figsize=(10,6))
#
# plt.plot(iterations, avg_total, label='Total active', color='black', linewidth=2)
# plt.plot(iterations, avg_red, label='Red nodes active', color='red', linestyle='--', linewidth=2)
# plt.plot(iterations, avg_blue, label='Blue nodes active', color='blue', linestyle='-.', linewidth=2)
#
# plt.xlabel('Iteration')
# plt.ylabel('Average number of newly active nodes')
# plt.title(f'Average spread per iteration')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # אחוזים
# total_nodes = len(graph.nodes())
# avg_red_percent = (sum(avg_red)/total_nodes) * 100
# avg_blue_percent = (sum(avg_blue)/total_nodes) * 100
#
# print(f"Average percentage of red nodes activated: {avg_red_percent:.2f}%")
# print(f"Average percentage of blue nodes activated: {avg_blue_percent:.2f}%")


def sigmoid(x, alpha=0.5):
    return 1 / (1 + np.exp(-alpha * x))

alpha = 0.1

# יצירת הסתברויות מהמשקלים
for u in graph.nodes():
    for v, _, data in graph.in_edges(u, data=True):
        data['prob'] = sigmoid(data['weight'], alpha)

# בחירת seed node באמצע טווח הדרגות
degrees = np.array([graph.degree(n) for n in graph.nodes()])
median_degree = np.median(degrees)
seed_node = min(graph.nodes(), key=lambda n: abs(graph.degree(n) - median_degree))

# סימולציית IC עם סטטיסטיקה
def run_ic_stats(graph, seed_node):
    active = set([seed_node])
    newly_active = set([seed_node])

    red_nodes = {n for n in graph.nodes() if graph.nodes[n].get("color") == "red"}
    blue_nodes = set(graph.nodes()) - red_nodes

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

    reason = "no new nodes" if len(active) < len(graph.nodes()) else "all nodes activated"
    return iter_counts, red_counts, blue_counts, reason

# הפעלת הסימולציה N פעמים
def influence_stats(graph, seed_node, n_sim=500):
    max_iters = 0
    all_iter_counts = []
    all_red_counts = []
    all_blue_counts = []
    reasons = []

    for _ in range(n_sim):
        iter_counts, red_counts, blue_counts, reason = run_ic_stats(graph, seed_node)
        max_iters = max(max_iters, len(iter_counts))
        all_iter_counts.append(iter_counts)
        all_red_counts.append(red_counts)
        all_blue_counts.append(blue_counts)
        reasons.append(reason)

    # ממוצע לכל איטרציה
    avg_total = np.zeros(max_iters)
    avg_red = np.zeros(max_iters)
    avg_blue = np.zeros(max_iters)

    for sim in range(n_sim):
        for i, val in enumerate(all_iter_counts[sim]):
            avg_total[i] += val / n_sim
        for i, val in enumerate(all_red_counts[sim]):
            avg_red[i] += val / n_sim
        for i, val in enumerate(all_blue_counts[sim]):
            avg_blue[i] += val / n_sim

    # אחוזים מצטברים
    cum_red_percent = np.cumsum(avg_red)/len(graph.nodes())*100
    cum_blue_percent = np.cumsum(avg_blue)/len(graph.nodes())*100

    avg_iters = np.mean([len(x) for x in all_iter_counts])
    reason_counts = {r: reasons.count(r) for r in set(reasons)}

    return avg_total, avg_red, avg_blue, cum_red_percent, cum_blue_percent, avg_iters, reason_counts

# שימוש
avg_total, avg_red, avg_blue, cum_red_percent, cum_blue_percent, avg_iters, reason_counts = influence_stats(graph, seed_node, n_sim=500)

# גרף מצטבר אחוזים
iterations = range(1, len(cum_red_percent)+1)
plt.figure(figsize=(10,6))
plt.plot(iterations, cum_red_percent, label='Red nodes cumulative %', color='red', linestyle='--', linewidth=2)
plt.plot(iterations, cum_blue_percent, label='Blue nodes cumulative %', color='blue', linestyle='-.', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cumulative % of activated nodes')
plt.title(f'Cumulative activation per iteration (seed node: {seed_node})')
plt.legend()
plt.grid(True)
plt.show()

print(f"Seed node (median degree): {seed_node}")
print(f"Average number of iterations: {avg_iters:.2f}")
print(f"Reasons for stopping:", reason_counts)

