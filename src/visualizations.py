# Imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns


def create_barcharts(results_dir, save_dir, dain_version=False):
    # Settings
    sns.set_style("white")

    main_color = "#f07f71" if not dain_version else "#EA9B2A"
    sat = 0.75 if not dain_version else 1

    # Read data
    df = pd.read_csv(results_dir)
    # Normalize probabilities
    df["Norm_prob"] = df["Prob"] / df.Category.map(
        df.groupby("Category")["Prob"].sum().to_dict()
    )
    # Format as percentage
    df["Norm_prob"] *= 100

    for cat in df.Category.unique():
        df_plot = df[df.Category == cat].copy()
        df_plot = df_plot.sort_values("Norm_prob", ascending=False)
        df_plot["Hundred_percent"] = 100

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 3))
        # Background (100%)
        sns.barplot(
            data=df_plot,
            x="Hundred_percent",
            y="Nominee",
            orient="h",
            saturation=sat,
            color="#d3d3d3",
        )
        # Plot probabilities
        sns.barplot(
            data=df_plot,
            x="Norm_prob",
            y="Nominee",
            orient="h",
            saturation=sat,
            color=main_color,
        )
        # Aesthetics
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        if dain_version:
            ax.spines["left"].set_color("#113341")
            ax.spines["right"].set_color("#113341")
            ax.spines["bottom"].set_color("#113341")
            ax.spines["top"].set_color("#113341")
            ax.yaxis.label.set_color("#113341")
            ax.xaxis.label.set_color("#113341")
            ax.tick_params(axis="x", colors="#113341")
            ax.tick_params(axis="y", colors="#113341")

        plt.xlim((0, 100))
        plt.ylabel("")
        ax.set_yticks(ax.get_yticks())
        if cat == "Picture":
            ax.set_yticklabels(df_plot["Film"])
        else:
            ax.set_yticklabels(list(df_plot["Nominee"] + " - " + df_plot["Film"]))
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(f"{save_dir}_{cat}.png")
        plt.close("all")
