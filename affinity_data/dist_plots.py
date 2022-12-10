from pathlib import Path
from affinity_data.analyze_affinity import FilterNames, \
    load_data_as_df
from data_handling.eval_funcs import eval_bleu_m2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statistics
import random
import numpy as np
import clize


def line_lens(row):
    return statistics.mean([len(row['ref'].split(" ")), len(row['pred'].split(" "))])


def len_stats(df):
    df['line_lens'] = df.apply(line_lens, axis=1)
    group_names = tuple(df['group_name'].unique())
    for group_name in group_names:
        print(f"Mean length of {group_name}")
        print(f"  {statistics.mean(df[df['group_name'] == group_name]['line_lens'])}")


def main(
    *,
    sample_size: int = 5000,
    seed: int = 42
):
    """
    Creating figure 5 in the paper.
    """
    random.seed(seed)
    np.random.seed(seed)
    print("Loading data")
    cur_file = Path(__file__).parent.absolute()
    df = load_data_as_df(
        Path(cur_file / "../data/affinity-data/affinity_fromfile1000-n.pkl.bz2").expanduser(),
        #Path("~/data/new-affinity-data/affinity_frompairs10.pkl.bz2").expanduser(),
        #Path("~/data/new-affinity-data/affinity_fromfile5.pkl.bz2").expanduser(),
        invalidate_cache=False,
        sample_size=sample_size
    )

    def func(row):
        ref, pred = row['ref'], row['pred']
        return eval_bleu_m2([ref], [pred])

    df['bleu'] = df.apply(func, axis=1)
    #print(df.head())

    plot_df = df[df['filter_name'] == FilterNames.AllFilts]
    #len_stats(plot_df)
    group_names = tuple(df['group_name'].unique())
    ax = sns.violinplot(
        x="group_name", y="bleu", #hue="filter_name",
        data=plot_df,
        cut=0,
        showmeans=True,
        inner="quartile",
    )
    ax.set(xlabel="Affinity Group", ylabel="BLEU-M2")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    for i, group_name in enumerate(group_names):
        gdf = plot_df[plot_df['group_name'] == group_name]
        # Draw mean lines
        mean = gdf['bleu'].mean()
        line_extent = 0.3
        ax.plot([i - line_extent, i + line_extent], [mean, mean], color="red", lw=3)
    plt.show()
    out_file = "dist_plots.png"
    print(f"Save to {out_file}")
    plt.savefig(out_file)


if __name__ == "__main__":
    clize.run(main)
