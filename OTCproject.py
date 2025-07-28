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
from networkx.algorithms.community import greedy_modularity_communities, modularity
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
matplotlib.use("TkAgg")
import random

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

import matplotlib.ticker as ticker

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

# settingOfSCCgraph(graph)
# plot_total_degree_distribution(graph)
# plot_degree_distribution_by_color(graph)
#powerLaw(graph)
#printInformOfBiggestSCCGraph(graph)
# pageRank(graph)


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

# settingOfSCCgraph(graph)
# plot_total_degree_distribution(graph)
# plot_degree_distribution_by_color(graph)


def FM(spreader_color, neighbor_color, P, Q):
    """
    Propagation probability function FM based on node colors.

    Parameters:
    - spreader_color: str, color of spreading node ('red' or 'blue')
    - neighbor_color: str, color of neighbor node
    - P: float, probability if colors are the same
    - Q: float, probability if colors are different

    Returns:
    - float: propagation probability
    """
    return P if spreader_color == neighbor_color else Q

def FU(node, FU_dict, default=0.5):
    """
    User adoption probability function FU.

    Parameters:
    - node: node ID
    - FU_dict: dict mapping node IDs to adoption probabilities
    - default: float, default adoption probability if node not in dict

    Returns:
    - float: adoption probability for the node
    """
    return FU_dict.get(node, default)

def FR(regulation_set_generator=None):
    """
    Regulation function FR generating a set of nodes to forcibly add as newly informed.

    Parameters:
    - regulation_set_generator: callable or None, returns set of nodes to add forcibly

    Returns:
    - set: nodes to add forcibly to currently newly informed nodes
    """
    if regulation_set_generator:
        return regulation_set_generator()
    return set()

def EC(history):
    """
    Compute Echo Chamber metrics over spreading history.

    Parameters:
    - history: list of dicts with counts of active nodes by color each step

    Returns:
    - list of dicts with keys:
      'step', 'total', 'red_fraction', 'blue_fraction'
    """
    ec_metrics = []
    for step, counts in enumerate(history):
        red = counts.get('red', 0)
        blue = counts.get('blue', 0)
        total = red + blue
        if total > 0:
            red_frac = red / total
            blue_frac = blue / total
        else:
            red_frac = blue_frac = 0.0
        ec_metrics.append({
            'step': step,
            'total': total,
            'red_fraction': red_frac,
            'blue_fraction': blue_frac
        })
    return ec_metrics

def spread_message_IC_with_all(G, seed_nodes, P, Q, FU_dict,
                               steps=10, regulation_set_generator=None):
    """
    Spread a message on graph G using Independent Cascade model combined with FM, FU, EC, FR.

    Parameters:
    - G: nx.DiGraph with 'color' attribute ('red'/'blue') on nodes
    - seed_nodes: list of initially active nodes
    - P: float, FM propagation probability for same color
    - Q: float, FM propagation probability for different color
    - FU_dict: dict {node: adoption probability}
    - steps: int, max propagation steps
    - regulation_set_generator: callable generating nodes forcibly added (FR), or None

    Returns:
    - history: list of dicts {color: count} cumulative over steps
    - ec_metrics: list of dicts with echo chamber data per step
    """
    informed = set(seed_nodes)
    newly_informed = set(seed_nodes)
    history = []

    for step in range(steps):
        next_newly_informed = set()
        for node in newly_informed:
            node_color = G.nodes[node]['color']
            for neighbor in G.neighbors(node):
                if neighbor in informed:
                    continue
                neighbor_color = G.nodes[neighbor]['color']
                prob = FM(node_color, neighbor_color, P, Q) * FU(neighbor, FU_dict)
                if random.random() < prob:
                    next_newly_informed.add(neighbor)

        # Apply regulation (FR)
        next_newly_informed |= FR(regulation_set_generator)

        if not next_newly_informed:
            break

        informed |= next_newly_informed
        newly_informed = next_newly_informed
        # Record count by color for EC
        color_counts = Counter(G.nodes[n]['color'] for n in informed)
        history.append(dict(color_counts))

    ec_metrics = EC(history)
    return history, ec_metrics

def plot_echo_chamber(ec_metrics):
    """
    Plot Echo Chamber fractions over time.

    Parameters:
    - ec_metrics: list of dicts with keys 'step', 'red_fraction', 'blue_fraction'
    """
    steps = [m['step'] for m in ec_metrics]
    red_frac = [m['red_fraction'] for m in ec_metrics]
    blue_frac = [m['blue_fraction'] for m in ec_metrics]

    plt.figure(figsize=(10,6))
    plt.plot(steps, red_frac, label='Red faction fraction', color='red')
    plt.plot(steps, blue_frac, label='Blue faction fraction', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Fraction of informed nodes')
    plt.title('Echo Chamber Dynamics Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

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
from networkx.algorithms.community import greedy_modularity_communities, modularity
import numpy as np
import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
matplotlib.use("TkAgg")
import random

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

import matplotlib.ticker as ticker

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
settingOfSCCgraph(graph)
# BehavioralHomophily(graph)
#printInformSourceGraph()
# plot_degree_distribution_by_color(graph)
#powerLaw(graph)
#printInformOfBiggestSCCGraph(graph)
# pageRank(graph)
# plot_total_degree_distribution(graph)

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

def print_spread_summary(G, informed_nodes, seed_node):
    """
    Prints a summary of the message spreading results:
    - The color of the starting node (seed)
    - How many nodes of each color were informed out of the total nodes of that color
    - How many nodes remain uninformed for each color (with counts and ratios)

    Parameters:
    - G: NetworkX graph
    - informed_nodes: set of nodes that received the message
    - seed_node: the seed node where spreading started
    """
    # Count total nodes by their 'color' attribute in the graph
    total_nodes_by_color = Counter(nx.get_node_attributes(G, 'color').values())
    # Count how many nodes of each color are in the informed set
    informed_colors = Counter(G.nodes[n]['color'] for n in informed_nodes)
    # Get the color of the seed node, or 'Unknown' if missing
    seed_color = G.nodes[seed_node].get('color', 'Unknown')

    print(f"Starting node: {seed_node} (color: {seed_color})")
    print("\nSpread summary by color:")
    for color, total_count in total_nodes_by_color.items():
        informed_count = informed_colors.get(color, 0)
        remaining = total_count - informed_count
        percent_informed = (informed_count / total_count * 100) if total_count > 0 else 0
        print(
            f"  Color '{color}': {informed_count} / {total_count} nodes informed ({percent_informed:.2f}%), remaining: {remaining}")

def FM(spreader_color, neighbor_color, P, Q):
    """
    Propagation probability function FM based on node colors.

    Parameters:
    - spreader_color: str, color of spreading node ('red' or 'blue')
    - neighbor_color: str, color of neighbor node
    - P: float, probability if colors are the same
    - Q: float, probability if colors are different

    Returns:
    - float: propagation probability
    """
    return P if spreader_color == neighbor_color else Q


def FU(node, FU_dict, default=0.5):
    """
    User adoption probability function FU.

    Parameters:
    - node: node ID
    - FU_dict: dict mapping node IDs to adoption probabilities
    - default: float, default adoption probability if node not in dict

    Returns:
    - float: adoption probability for the node
    """
    return FU_dict.get(node, default)


def FR(regulation_set_generator=None, informed=None):
    """
    Regulation function FR generating a set of nodes to forcibly add as newly informed.

    Parameters:
    - regulation_set_generator: callable or None, returns set of nodes to add forcibly

    Returns:
    - set: nodes to add forcibly to currently newly informed nodes
    """
    if regulation_set_generator:
        return regulation_set_generator(informed)
    return set()


def EC(history):
    """
    Compute Echo Chamber metrics over spreading history.

    Parameters:
    - history: list of dicts with counts of active nodes by color each step

    Returns:
    - list of dicts with keys:
      'step', 'total', 'red_fraction', 'blue_fraction'
    """
    ec_metrics = []
    for step, counts in enumerate(history):
        red = counts.get('red', 0)
        blue = counts.get('blue', 0)
        total = red + blue
        if total > 0:
            red_frac = red / total
            blue_frac = blue / total
        else:
            red_frac = blue_frac = 0.0
        ec_metrics.append({
            'step': step,
            'total': total,
            'red_fraction': red_frac,
            'blue_fraction': blue_frac
        })
    return ec_metrics


def spread_message_IC_with_all(G, seed_nodes, P, Q, FU_dict,
                               steps=10, regulation_set_generator=None):
    """
    Spread a message on graph G using Independent Cascade model combined with FM, FU, EC, FR.

    Parameters:
    - G: nx.DiGraph with 'color' attribute ('red'/'blue') on nodes
    - seed_nodes: list of initially active nodes
    - P: float, FM propagation probability for same color
    - Q: float, FM propagation probability for different color
    - FU_dict: dict {node: adoption probability}
    - steps: int, max propagation steps
    - regulation_set_generator: callable generating nodes forcibly added (FR), or None

    Returns:
    - history: list of dicts {color: count} cumulative over steps
    - ec_metrics: list of dicts with echo chamber data per step
    - informed: set of all nodes that received the message by end
    """
    informed = set(seed_nodes)
    newly_informed = set(seed_nodes)
    history = []

    for step in range(steps):
        next_newly_informed = set()
        for node in newly_informed:
            node_color = G.nodes[node]['color']
            for neighbor in G.neighbors(node):
                if neighbor in informed:
                    continue
                neighbor_color = G.nodes[neighbor]['color']
                prob = FM(node_color, neighbor_color, P, Q) * FU(neighbor, FU_dict)
                if random.random() < prob:
                    next_newly_informed.add(neighbor)

        # Apply regulation (FR)
        next_newly_informed |= FR(regulation_set_generator, informed)

        if not next_newly_informed:
            break

        informed |= next_newly_informed
        newly_informed = next_newly_informed
        # Record count by color for EC
        color_counts = Counter(G.nodes[n]['color'] for n in informed)
        history.append(dict(color_counts))

    ec_metrics = EC(history)
    return history, ec_metrics, informed

red_nodes = [n for n in graph.nodes() if graph.nodes[n].get("color") == "red"]
sorted_red_nodes = sorted(red_nodes, key=lambda n: graph.in_degree(n) + graph.out_degree(n), reverse=True)

seed_nodes = [sorted_red_nodes[0]]

def example_FR(informed):
    uninformed_nodes = list(set(graph.nodes()) - informed)
    if not uninformed_nodes:
        return set()
    num_to_add = int(len(informed) * 0.05)
    num_to_add = min(num_to_add, len(uninformed_nodes))
    return set(random.sample(uninformed_nodes, num_to_add))


num_runs = 30
steps = 15

all_histories = []
all_ec_metrics = []
all_informed_counts = []
informed = 0

for _ in range(num_runs):
    FU_dict = {node: random.uniform(0.2, 0.6) for node in graph.nodes()}
    history, ec_metrics, informed = spread_message_IC_with_all(
        graph,
        seed_nodes,
        P=0.7,
        Q=0.3,
        FU_dict=FU_dict,
        steps=steps,
        regulation_set_generator=example_FR
    )
    all_histories.append(history)
    all_ec_metrics.append(ec_metrics)
    all_informed_counts.append([sum(step.values()) for step in history])


max_len = max(len(run) for run in all_informed_counts)

padded_counts = [
    run + [run[-1]] * (max_len - len(run)) if run else [0] * max_len
    for run in all_informed_counts
]

avg_informed_per_step = np.mean(padded_counts, axis=0)
steps_range = range(len(avg_informed_per_step))

plt.figure(figsize=(10, 6))
plt.plot(steps_range, avg_informed_per_step, label='Average informed', color='blue', linewidth=2)

plt.xlabel("Step")
plt.ylabel("Number of informed nodes")
plt.title("Average message spread over 30 runs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print_spread_summary(graph, informed, seed_nodes[0])