# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Settings
sns.set_style("white")


def create_barcharts(new_season, model, dain_version=False):
    main_color = "#f07f71" if not dain_version else "#EA9B2A"
    sat = 0.75 if not dain_version else 1

    # Read data
    df = pd.read_csv(f"results/all_predictions_{model}_{new_season}.csv")
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
        if cat == "Picture":
            ax.set_yticklabels(df_plot["Film"])
        else:
            ax.set_yticklabels(list(df_plot["Nominee"] + " - " + df_plot["Film"]))
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(f"results/predictions_barchart_{model}_{new_season}_{cat}.png")
        plt.close("all")


create_barcharts(new_season="2024", model="logit", dain_version=False)
