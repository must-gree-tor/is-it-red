import is_it_red as iir

import matplotlib.pyplot as plt

# Define a recursive function to plot the tree
def plot_tree(node, x=0.5, y=1, dx=0.2, dy=0.1, depth=0, max_depth=4):
    if isinstance(node, dict):
        # Plot the current node
        plt.text(x, y, f"Index: {node['index']}\nValue: {node['value']}\nScore: {node['score']:.3f}", 
                 ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))

        # Plot left child if it exists
        if 'left' in node:
            next_x = x - dx / (2 ** depth)
            next_y = y - dy
            plt.plot([x, next_x], [y, next_y], color='black')
            plot_tree(node['left'], x=next_x, y=next_y, dx=dx, dy=dy, depth=depth+1)

        # Plot right child if it exists
        if 'right' in node:
            next_x = x + dx / (2 ** depth)
            next_y = y - dy
            plt.plot([x, next_x], [y, next_y], color='black')
            plot_tree(node['right'], x=next_x, y=next_y, dx=dx, dy=dy, depth=depth+1)
    else:
        # Plot leaf node
        plt.text(x, y, f"Leaf: {node}", ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))

# Set up the plot
plt.figure(figsize=(8, 6))
plt.title("Decision Tree Visualization")
plt.axis('off')  # Turn off the axis

plot_tree(iir.tree)

plt.show()