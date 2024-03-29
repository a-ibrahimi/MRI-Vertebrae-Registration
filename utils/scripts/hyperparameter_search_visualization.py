"""
This script visualizes hyperparameter search results after running hyper_search.sh. It will pick up the hyperparameters
and make a plot. The best hyperparameter setting will be marked with a red star and the loss function will be labeled accordingly.
This will save time and effort in finding the best hyperparameters for the model. It can be easily extended to include more
hyperparameters by adjusting the pattern or regex. The script can be run from the command line or imported as a module.
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_hyperparameters(file_path):
    """
    Reads a text file containing hyperparameter search results,
    parses the data, creates visualizations in a single row,
    adds grid lines to each plot, and highlights the point
    with the lowest Average Dice Score along with its loss function.

    Parameters:
    file_path (str): Path to the text file containing the hyperparameter search results.
    """
    with open(file_path, 'r') as file:
        file_content = file.read()

    data = []
    pattern = r"Loss: (\w+), Lambda: ([\d.]+), Gamma: ([\d.]+), Learning rate: ([\de.-]+)\s+Average dice score:\s+([-\d.]+)"

    for line in file_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            loss, lambda_, gamma, lr, dice_score = match.groups()
            data.append({
                "Loss": loss,
                "Lambda": float(lambda_),
                "Gamma": float(gamma),
                "Learning Rate": float(lr),
                "Average Dice Score": float(dice_score)
            })

    df = pd.DataFrame(data)

    # Find the point with the lowest Average Dice Score
    min_dice_row = df.loc[df['Average Dice Score'].idxmin()]

    # Visualization
    plt.figure(figsize=(18, 6))

    # Lambda vs. Average Dice Score
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='Lambda', y='Average Dice Score', hue='Loss')
    plt.scatter(min_dice_row['Lambda'], min_dice_row['Average Dice Score'], color='red', marker='*', s=100) # Highlighting with star
    plt.text(min_dice_row['Lambda'], min_dice_row['Average Dice Score'], f" {min_dice_row['Loss']}", color='red') # Labeling Loss function
    plt.title('Lambda vs. Average Dice Score')
    plt.grid(True) # Adding grid lines

    # Gamma vs. Average Dice Score
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='Gamma', y='Average Dice Score', hue='Loss')
    plt.scatter(min_dice_row['Gamma'], min_dice_row['Average Dice Score'], color='red', marker='*', s=100) 
    plt.text(min_dice_row['Gamma'], min_dice_row['Average Dice Score'], f" {min_dice_row['Loss']}", color='red') 
    plt.title('Gamma vs. Average Dice Score')
    plt.grid(True) 

    # Learning Rate vs. Average Dice Score
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='Learning Rate', y='Average Dice Score', hue='Loss')
    plt.scatter(min_dice_row['Learning Rate'], min_dice_row['Average Dice Score'], color='red', marker='*', s=100) 
    plt.text(min_dice_row['Learning Rate'], min_dice_row['Average Dice Score'], f" {min_dice_row['Loss']}", color='red') 
    plt.xscale('log')  
    plt.title('Learning Rate vs. Average Dice Score')
    plt.grid(True) 

    plt.tight_layout()
    plt.show()

    best_hyperparameters = {
        "Loss": min_dice_row['Loss'],
        "Lambda": min_dice_row['Lambda'],
        "Gamma": min_dice_row['Gamma'],
        "Learning Rate": min_dice_row['Learning Rate'],
        "Average Dice Score": min_dice_row['Average Dice Score']
    }

    print("Best Hyperparameter Combination:")
    for key, value in best_hyperparameters.items():
        print(f"{key}: {value}")