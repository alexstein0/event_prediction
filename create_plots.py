import os
import re

import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


def combine_latest_results(directory, phase):
    # Regular expression pattern to match the filenames
    pattern = re.compile(r'table-(.+)(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})-(train|eval)_stats\.csv')

    # Function to parse the filename and extract information
    def parse_filename(filename):
        match = pattern.match(filename)
        if match:
            task_experiment_part = match.group(1)[:-1]  # remove splitter
            creation_time_str = match.group(2)
            train_eval = match.group(3)

            # Split task_experiment_part by the last hyphen
            splitted = task_experiment_part.split('_')
            task_name = "_".join(splitted[:3])
            experiment_name = "_".join(splitted[3:])
            if len(experiment_name) <= 1:
                print("no task name")
                experiment_name = "base"

            # Parse creation time
            creation_time = datetime.strptime(creation_time_str, "%Y-%m-%d-%H-%M")

            return task_name, experiment_name, creation_time, train_eval
        else:
            raise ValueError(f"Filename format incorrect: {filename}")

    # Read all CSV files from the specified directory and store their info in a list
    file_info = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            try:
                task_name, experiment_name, creation_time, train_eval = parse_filename(filename)
                print(task_name, experiment_name, creation_time, train_eval)
                if train_eval != phase:
                    continue
                file_info.append((task_name, experiment_name, creation_time, train_eval, filename))
            except ValueError as e:
                print(e)  # Print error message and skip incorrect files

    # Convert the file info into a DataFrame
    file_df = pd.DataFrame(file_info,
                           columns=['task_name', 'experiment_name', 'creation_time', 'train_eval', 'filename'])
    print(file_df)  # Debug print

    # Filter to keep only the latest creation time for each task and experiment
    latest_files = file_df.loc[file_df.groupby(['task_name', 'experiment_name'])['creation_time'].idxmax()]
    print(latest_files.columns)

    # Filter further based on the phase (train or eval)
    latest_phase_files = latest_files[latest_files['train_eval'] == phase]
    print(latest_phase_files.columns)
    print(latest_phase_files)

    # Combine the filtered CSV files into a single DataFrame
    combined_df = pd.concat([pd.read_csv(os.path.join(directory, f), sep='\t') for f in latest_phase_files['filename']])
    print(combined_df)
    split_col = combined_df["name"].str.split('_')
    selected_parts = split_col.apply(lambda x: '_'.join(x[3:]))
    selected_parts[selected_parts == ""] = "Base"
    combined_df["experiment_name"] = selected_parts

    # Reset the index of the combined DataFrame
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined DataFrame to a new CSV file (optional)
    save_loc = f'combined_{phase}_results'
    combined_df.to_csv(os.path.join(dir, f"{save_loc}.csv"), sep='\t', index=False)
    print(combined_df.columns)
    plot(combined_df, save_path=os.path.join(dir, f"{save_loc}.png"))


def plot(combined_df, save_path=None, dataset=None):

    # Grouped bar chart

    combined_df['eval_setup'] = combined_df.apply(lambda row: f'randomize={row.randomize_order}/mask={row.masking}', axis=1)
    combined_df = combined_df.drop_duplicates(subset=['saved_name', 'eval_setup'], keep='first')

    # Print the combined DataFrame (or perform further processing)
    dataset = "electronics"
    electronics = {
        "dataset": "amazon_movies_5core",
        "TabBERT": .7964,
        "FATA-TRANS": .8057,
    }
    movies = {
        "dataset": "amazon_electronics_5core",
        "TabBERT": .7098,
        "FATA-TRANS": .7206,
    }
    ibm = {
        "dataset": "ibm_fraud_transaction",
        "TabBERT": .9985,
        "FATA-TRANS": .9992,
    }
    baselines = {
        "electronics": electronics,
        "movies": movies,
        "ibm": ibm,
    }

    eval_setup = f"randomize=False/mask=False"

    # combined_df = combined_df[eval_setup == combined_df['eval_setup']]
    # group_by_col = "data_name"
    name = "experiment_name"
    value_col = "eval_auc_last_consolidated"

    group_by_col = "eval_setup"
    combined_df = combined_df[baselines[dataset]["dataset"] == combined_df['data_name']]
    cols = ["name", "experiment_name", "eval_auc", "eval_auc_last", "data_name", "eval_auc_consolidated", "eval_auc_last_consolidated", "eval_setup"]
    cols2 = ["name", "experiment_name", "eval_auc", "eval_auc_last", "data_name", "eval_setup"]

    try:
        data = combined_df[cols]
    except:
        data = combined_df[cols2]
        value_col = "eval_auc_last"

    plt.figure(figsize=(12, 24), dpi=300)
    sns.set_theme(rc={"figure.dpi": 600})

    g = sns.catplot(
        x=group_by_col,  # x variable name
        y=value_col,  # y variable name
        hue=name,  # group variable name
        data=data,  # dataframe to plot
        kind="bar",
        height=6,
        # legend=False,  # Remove the legend
    )
    # Add horizontal line and adjust y-axis limits if necessary
    y = baselines[dataset]["FATA-TRANS"]
    for ax in g.axes.flat:
        ax.axhline(y=y, color='r', linestyle='--', label="FATA-TRANS")
        # ax.set_ylim(0, 1)  # Adjust this range if needed

    # Set x and y labels
    plt.title(f"Performance on {dataset}", fontsize=48)
    g.set_xlabels("Testing Regime")
    g.set_ylabels("AUC last (consolidated)")
    # sns.move_legend(g, bbox_to_anchor=(0, .2), loc='lower left', title='Training Regime')

    sns.move_legend(g, loc='upper right', title='Training Regime')
    g.set_xticklabels(rotation=20)
    g.set(ylim=(y-.25, min(.9, y+.25)))
    # g.set(ylim=(.95, 1.0))

    plt.xticks(rotation=35)
    plt.tight_layout()

    if save_path is not None:
        print(f"save {save_path}")
        g.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Example usage
    ds = "EVAL_final_amazon"
    # ds = "EVAL_final_IBM_other"

    dir = f'/cmlscratch/astein0/event_prediction/outputs/{ds}'
    dir = f"/Users/alex/Documents/School/Maryland/Research/event_prediction/outputs/{ds}"
    train_eval = "eval"
    combine_latest_results(dir, train_eval)

    csv_path = f"/Users/alex/Documents/School/Maryland/Research/event_prediction/outputs/{ds}/combined_eval_results.csv"
    save_loc = f'combined_{train_eval}_results.png'
    # plot(pd.read_csv(csv_path, sep='\t'), os.path.join(dir, save_loc))


    # present data
    data_path = "/Users/alex/Documents/School/Maryland/Research/event_prediction/data/submit"
    tables = ["raw.csv", "split.csv", "tokens.csv", "batch.csv"]
    for t in tables:
        path = os.path.join(data_path, t)
        df = pd.read_csv(path)
        print(df)

    def process_line(line):
        # Split the line by commas
        parts = line.split(',')
        # Process each part
        new_parts = []
        for part in parts:
            # Check if the part is a number and has less than 3 digits
            if part.isdigit() and len(part) < 4:
                new_parts.append(part + '\t')
            else:
                new_parts.append(part)
        # Join the parts back into a string
        return ''.join(new_parts)

    data = [
        "1,8,6,100,10,5,7,89,18,6,5,793,29,6,10,5,7,17,116,6,5,6,22,117,7,5,7,609,6,28,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,7,6,500,8,5,703,8,6,11,5,7,6,86,25,5,6,102,11,8,5,7,8,748,6,5,11,6,8,33,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,121,27,7,9,5,299,12,8,9,5,16,12,272,9,5,14,209,12,9,5,7,9,8,217,5,10,9,29,436,5,9,376,13,17,5,9,14,313,7,5,11,265,8,9,5,910,26,11,9,5,2",
        "1,880,9,11,20,5,11,9,243,14,5,9,18,10,127,5,303,14,13,9,5,14,360,9,7,5,11,309,9,14,5,14,7,133,9,5,8,9,13,264,5,12,16,9,245,5,192,9,8,7,5,2",
        "1,8,9,390,12,5,275,10,9,14,5,9,8,7,274,5,9,125,8,12,5,9,355,26,11,5,9,399,11,8,5,289,9,14,10,5,190,9,15,10,5,21,10,407,9,5,8,9,364,10,5,2",
        "1,9,10,29,160,5,9,404,8,10,5,16,277,9,10,5,16,409,9,11,5,9,418,10,8,5,15,10,426,9,5,8,424,11,9,5,77,7,6,14,5,77,7,6,29,5,0,0,0,0,0,2",
        "1,332,6,7,8,5,8,7,6,291,5,7,6,8,716,5,7,139,8,6,5,6,22,7,681,5,6,7,8,363,5,28,734,7,6,5,325,6,7,15,5,6,472,8,7,5,6,485,8,7,5,2",
        "1,9,15,7,142,5,207,9,7,23,5,7,240,14,9,5,12,8,6,47,5,10,25,241,9,5,19,340,12,9,5,7,177,9,14,5,9,403,7,19,5,25,379,9,10,5,11,40,9,24,5,2",
        "1,9,21,7,531,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,21,7,410,9,5,7,14,6,483,5,9,7,8,83,5,15,7,83,9,5,9,10,8,346,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,95,6,7,8,5,27,31,6,7,5,7,539,6,20,5,17,7,6,870,5,6,7,116,17,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,28,600,7,6,5,8,9,385,7,5,6,7,21,683,5,9,559,8,7,5,6,8,7,428,5,27,7,652,6,5,6,462,14,7,5,9,7,490,14,5,6,173,7,8,5,7,606,6,21,5,2",
        "1,8,417,6,7,5,8,10,6,487,5,7,6,8,347,5,7,9,18,92,5,8,9,7,397,5,17,7,455,9,5,7,25,9,516,5,688,7,6,8,5,869,6,7,8,5,863,28,7,6,5,2",
        "1,6,7,8,225,5,223,7,16,6,5,6,7,8,135,5,7,6,8,615,5,7,6,8,56,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        "1,6,618,13,8,5,6,7,115,8,5,6,32,8,7,5,119,6,21,7,5,8,7,6,902,5,8,36,6,7,5,7,38,17,6,5,6,16,7,35,5,7,6,263,17,5,11,8,6,48,5,2",
        "1,6,45,11,8,5,8,11,42,6,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2",
        # Add more lines as needed...
    ]

    # Process each line
    processed_data = [process_line(line) for line in data]

    # Print the processed data
    for line in processed_data:
        print(line)

    # create table
    csv_path = f"/Users/alex/Documents/School/Maryland/Research/event_prediction/outputs/{ds}/combined_eval_results.csv"
    df = pd.read_csv(csv_path, sep='\t')

    print(df)
    df = df[df["data_name"] == 'amazon_movies_5core']
    base = (df[df["experiment_name"]=="transaction-base"]).pivot(index="randomize_order", columns="masking", values="eval_auc")
    both = (df[df["experiment_name"]=="transaction-masked_randomized"]).pivot(index="randomize_order", columns="masking", values="eval_auc")
    print(base)
    print(both)

    metric = "eval_auc_consolidated_last" if "eval_auc_consolidated_last" in df.columns else "eval_auc_last"
    relevant = df[["randomize_order", "masking", "experiment_name", metric,]]
    keyed = relevant.sort_values(["randomize_order", "masking", "experiment_name"])
    print(keyed)

