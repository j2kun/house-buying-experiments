from scipy.stats import describe
from scipy.stats import gaussian_kde
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from model import Model
from model import ExperimentParameters
from model import run_experiment


def fit_kde(data):
    snp_percent_change = data["Percent change"].to_numpy()
    kde = gaussian_kde(snp_percent_change)
    return Model(name="Kernel Density Estimator",
                 sampler=lambda n: list(kde.resample(size=n)[0]))


if __name__ == "__main__":
    data = pd.read_csv("snp.tsv", sep='\t')
    model = fit_kde(data.head(50 * 12))

    params = ExperimentParameters(
        years=30,
        house_price_usd=1000000,
        on_hand_usd=700000,
        mortgage_rate=0.03,
    )

    results = run_experiment(model, params)
    df = pd.DataFrame(data=results)

    fig, axes = plt.subplots()
    sns.violinplot(data=df, ax=axes, scale="width")
    # axes = df.boxplot()
    axes.set_title("Larger down payment vs Investing in S&P 500")
    axes.set_xlabel(
        f"Down payment ({params.on_hand_usd / 1000}k - x is invested)")
    axes.set_ylabel(f"{params.years} year profit minus loan interest")
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%d"))
    axes.yaxis.grid(True)

    plt.tight_layout()
    plt.show()
