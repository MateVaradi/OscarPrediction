# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings
sns.set_style("white")

def create_barcharts(new_season):
    # Read data
    df = pd.read_csv(f'results/all_predictions_{new_season}.csv')
    # Normalize probabilities
    df['Norm_prob'] = df['Prob'] / df.Category.map(df.groupby('Category')['Prob'].sum().to_dict())
    # Format as percentage
    df['Norm_prob'] *= 100

    for cat in df.Category.unique():
        df_plot = df[df.Category == cat].copy()
        df_plot = df_plot.sort_values('Norm_prob', ascending=False)
        df_plot['Hundred_percent'] = 100

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 3))
        # Background (100%)
        sns.barplot(data=df_plot, x='Hundred_percent',y='Nominee', orient='h', color='#d3d3d3')
        # Plot probabilities
        sns.barplot(data=df_plot, x='Norm_prob', y='Nominee', orient='h', color='#f07f71')
        # Aesthetics
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xlim((0, 100))
        plt.ylabel('')
        if cat == 'Picture':
            ax.set_yticklabels(df_plot['Film'])
        else:
            ax.set_yticklabels(list(df_plot['Nominee'] + ' - ' +df_plot['Film']))
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(f'results/predictions_barchart_{new_season}_{cat}.png')
        plt.close('all')

create_barcharts('2020')