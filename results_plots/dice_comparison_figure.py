import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_pathes", help="pathes to results",
                        type=str, required=True)
    parser.add_argument("--column_names", help="column names to extract for comparison",
                        type=str, required=True)
    return parser.parse_args()


def plot_results(df):
    plt.figure(figsize=(8, 6))
    specific_colors = ['#FF5733', '#8da0cb', '#33FFCE']
    # Use seaborn's boxplot, grouped by 'Group' and colored by 'Method'
    sns.set_context("talk", font_scale=1.005)
    sns.boxplot(x='Group', y='Score', hue='Method', data=df,
                palette=specific_colors,  # You can choose your palette
                showfliers=True)  # Show outliers if you want

    plt.title('MAE for Dice scores estimation')
    plt.ylabel('MAE')  # or your own metric label
    plt.ylim(0, 0.6)  # adjust to your data range

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')

    plt.tight_layout()
    plt.show()


def parse_data(results_pathes, column_names):
    pathes = results_pathes.split(',')
    column_names = column_names.split(',')

    dfs = []
    for path in pathes:
        df = pd.read_csv(path)
        df = df[column_names]
        df_melted = df.melt(var_name='Method', value_name='Score')
        df_melted["Group"] = os.path.basename(path)[14:-4]
        dfs.append(df_melted)
    unified_df = pd.concat(dfs)
    return unified_df


if __name__ == "__main__":
    opts = get_arguments()
    data_df = parse_data(opts.results_pathes, opts.column_names)
    plot_results(data_df)