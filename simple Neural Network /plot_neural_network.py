import matplotlib.pyplot as plt
import networkx as nx

# Define the structure of the neural network with biases
def draw_neural_network_with_bias():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for input layer
    input_nodes = ["I1", "I2"]
    G.add_nodes_from(input_nodes)

    # Add bias node for the hidden layer
    G.add_node("B1")

    # Add nodes for hidden layer
    hidden_nodes = ["H1", "H2"]
    G.add_nodes_from(hidden_nodes)

    # Add bias node for the output layer
    G.add_node("B2")

    # Add nodes for output layer
    output_nodes = ["O1", "O2"]
    G.add_nodes_from(output_nodes)

    # Add edges between input and hidden layers
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            G.add_edge(input_node, hidden_node)

    # Add edges from bias B1 to hidden layer
    for hidden_node in hidden_nodes:
        G.add_edge("B1", hidden_node)

    # Add edges between hidden and output layers
    for hidden_node in hidden_nodes:
        for output_node in output_nodes:
            G.add_edge(hidden_node, output_node)

    # Add edges from bias B2 to output layer
    for output_node in output_nodes:
        G.add_edge("B2", output_node)

    # Position the nodes for visualization
    pos = {
        "I1": (-1, 1), "I2": (-1, -1),
        "B1": (-0.5, 0),  # Bias for hidden layer
        "H1": (0, 1), "H2": (0, -1),
        "B2": (0.5, 0),   # Bias for output layer
        "O1": (1, 1), "O2": (1, -1)
    }

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=2000, node_color="lightblue",
        font_size=16, font_weight="bold", arrowsize=20,
        edgecolors="black"  # Add borders to nodes for better visibility
    )
    plt.title("Neural Network Structure with Biases", fontsize=20)
    plt.show()

if __name__ == "__main__":
    draw_neural_network_with_bias()