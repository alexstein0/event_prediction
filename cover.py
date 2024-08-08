# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
#
# # Function to create a custom shape with the top right corner cut off
# def create_custom_card(x, y, width, height, color):
#     vertices = [
#         (x - width / 2, y - height / 2),  # bottom-left
#         (x + width / 2, y - height / 2),  # bottom-right
#         (x + width / 2, y + height / 2 - 0.1),  # top-right (cut-off)
#         (x + width / 2 - 0.1, y + height / 2),  # top-right (cut-off)
#         (x - width / 2, y + height / 2),  # top-left
#         (x - width / 2, y - height / 2)   # back to bottom-left
#     ]
#     codes = [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.CLOSEPOLY]
#     path = patches.Path(vertices, codes)
#     patch = patches.PathPatch(path, facecolor=color, edgecolor='black', linewidth=1.5)
#     return patch
#
# # Create figure
# fig, ax = plt.subplots(figsize=(12, 4))
#
# # Define positions and details
# x_positions = np.linspace(1, 5, 5)
# y_position = 0.5
# states = ['NY', 'NY', 'CA', 'NY', 'NY']
# timestamps = [f'07-{i+1}-24' for i in range(5)]
# dollar_amounts = np.random.randint(50, 100, size=5)
# is_fraud = ['False', 'False', 'True', 'False', '?']
# colors = ['#a6cee3' if fraud == 'False' else '#fb9a99' if fraud == 'True' else '#fdbf6f' for fraud in is_fraud]
#
# # Plot each transaction
# for i, (x, state, timestamp, amount, fraud, color) in enumerate(zip(x_positions, states, timestamps, dollar_amounts, is_fraud, colors)):
#     card_box = create_custom_card(x, y_position, 0.8, 0.3, color)
#     ax.add_patch(card_box)
#     plt.text(x - 0.35, y_position + 0.1, f'STATE: {state}', ha="left", va="center", fontsize=9)
#     plt.text(x - 0.35, y_position + 0.025, f'TIMESTAMP: {timestamp}', ha="left", va="center", fontsize=9)
#     plt.text(x - 0.35, y_position - 0.05, f'AMOUNT: ${amount}', ha="left", va="center", fontsize=9)
#     if fraud == '?':
#         plt.text(x - 0.35, y_position - 0.125, f'IS FRAUD: ', ha="left", va="center", fontsize=9)
#         plt.text(x, y_position - 0.125, f'{fraud}', ha="left", va="center", fontsize=9, color='red', weight='bold')
#     else:
#         plt.text(x - 0.35, y_position - 0.125, f'IS FRAUD: {fraud}', ha="left", va="center", fontsize=9)
#
# # Add arrows to connect the transactions
# for i in range(len(x_positions) - 1):
#     ax.annotate('', xy=(x_positions[i + 1] - 0.5, y_position), xytext=(x_positions[i] + 0.4, y_position),
#                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
#
# # Add a timeline at the bottom
# plt.plot([0.6, 5.4], [0.25, 0.25], color='black', lw=1.5)
# for i, x in enumerate(x_positions):
#     plt.text(x, 0.2, f'T{i + 1}', ha='center', va='center', fontsize=10)
#     plt.plot([x, x], [0.23, 0.27], color='black', lw=1.5)
#
# # Set limits and hide axes
# ax.set_xlim(0.5, 5.5)
# ax.set_ylim(0, 1)
# ax.axis('off')
#
# # Add title
# plt.title("Event Prediction in Credit Card Transactions", fontsize=14, weight='bold')
#
# # Show the plot
# plt.show()
#


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import random

# Create a DataFrame with 5 sets of 5 transactions each
data = {
    'Set': np.repeat(np.arange(1, 6), 5),
    'Transaction': np.tile(np.arange(1, 6), 5),
    'State': np.random.choice(['NY', 'CA', 'TX', 'FL'], size=25),
    'Timestamp': pd.date_range(start='2024-07-01', periods=25, freq='D').strftime('%d-%m-%Y'),
    'Amount': np.random.randint(50, 100, size=25),
    'Is_Fraud': np.random.choice([True, False, '?'], size=25, p=[0.1, 0.8, 0.1])
}

df = pd.DataFrame(data)
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to create a custom shape with the top right corner cut off
def create_custom_card(x, y, width, height, color):
    vertices = [
        (x - width / 2, y - height / 2),  # bottom-left
        (x + width / 2, y - height / 2),  # bottom-right
        (x + width / 2, y + height / 2 - 0.1),  # top-right (cut-off)
        (x + width / 2 - 0.1, y + height / 2),  # top-right (cut-off)
        (x - width / 2, y + height / 2),  # top-left
        (x - width / 2, y - height / 2)  # back to bottom-left
    ]
    codes = [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO,
             patches.Path.CLOSEPOLY]
    path = patches.Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor=color, edgecolor='black', linewidth=1.5)
    return patch


# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Define y positions for each sequence
y_positions = np.linspace(1, 10, 5)

# Iterate over each sequence
for set_num, y_position in enumerate(y_positions, start=1):
    subset = df[df['Set'] == set_num]
    x_positions = np.linspace(1, 5, 5)

    # Plot each transaction
    for i, (index, row) in enumerate(subset.iterrows()):
        x = x_positions[i]
        state = row['State']
        timestamp = row['Timestamp']
        amount = row['Amount']
        fraud = row['Is_Fraud']
        color = '#a6cee3' if fraud == 'False' else '#fb9a99' if fraud == 'True' else '#fdbf6f'

        card_box = create_custom_card(x, y_position, 0.8, 0.3, color)
        ax.add_patch(card_box)
        plt.text(x - 0.35, y_position + 0.1, f'STATE: {state}', ha="left", va="center", fontsize=9)
        plt.text(x - 0.35, y_position + 0.025, f'TIMESTAMP: {timestamp}', ha="left", va="center", fontsize=9)
        plt.text(x - 0.35, y_position - 0.05, f'AMOUNT: ${amount}', ha="left", va="center", fontsize=9)
        if fraud == '?':
            plt.text(x - 0.35, y_position - 0.125, f'IS FRAUD: ', ha="left", va="center", fontsize=9)
            plt.text(x, y_position - 0.125, f'{fraud}', ha="left", va="center", fontsize=9, color='red', weight='bold')
        else:
            plt.text(x - 0.35, y_position - 0.125, f'IS FRAUD: {fraud}', ha="left", va="center", fontsize=9)

    # Add arrows to connect the transactions
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i + 1] - 0.5, y_position), xytext=(x_positions[i] + 0.4, y_position),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))

    # Add a timeline at the bottom of each sequence
    plt.plot([0.6, 5.4], [y_position - 0.25, y_position - 0.25], color='black', lw=1.5)
    for i, x in enumerate(x_positions):
        plt.text(x, y_position - 0.3, f'T{i + 1}', ha='center', va='center', fontsize=10)
        plt.plot([x, x], [y_position - 0.27, y_position - 0.23], color='black', lw=1.5)

# Set limits and hide axes
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0, 11)
ax.axis('off')

# Add title
plt.title("Event Prediction in Credit Card Transactions", fontsize=14, weight='bold')

# Show the plot
plt.show()