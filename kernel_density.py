from scipy.stats import describe
from scipy.stats import gaussian_kde
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


def learn_model(data):
    snp_percent_change = data["Percent change"].to_numpy()
    kde = gaussian_kde(snp_percent_change)
    return lambda n: kde.resample(size=n)[0]


def run_experiment(sampler, years=15, capital=400000):
    num_months = years * 12
    starting_capital = capital
    samples = sampler(num_months)

    for monthly_pct_change in samples:
        capital *= (1 + monthly_pct_change / 100)
        if capital <= 0:
            return starting_capital

    return capital - starting_capital


def run_experiments(sampler, years=15, capital=400000, num_runs=1000):
    results = []
    for i in range(num_runs):
        results.append(run_experiment(sampler, years=years, capital=capital))
    return results


def compute_loan_interest(loan_amt, yearly_rate, years=15):
    return loan_amt * (1 + yearly_rate / 12)**(12 * years) - loan_amt


if __name__ == "__main__":
    data = pd.read_csv("snp.tsv", sep='\t')
    sampler = learn_model(data.head(50*12))

    house_price = 1000000
    cash_on_hand = 700000
    mortgage_rate = 0.03
    down_payments = [200000 + i * 50000 for i in range(10)]
    loan_interests = dict(
        (dp, compute_loan_interest(house_price - dp, mortgage_rate))
        for dp in down_payments)
    capital_values = dict((dp, cash_on_hand-dp) for dp in down_payments)

    results_dict = dict()
    for dp, capital in capital_values.items():
        print("Simulating for $%.2f starting capital" % capital)
        results = run_experiments(sampler, capital=capital)
        profits_minus_loan_interest = [
                profit - loan_interests[dp] for profit in results]
        results_dict["{:d}k".format(dp // 1000)] = profits_minus_loan_interest

    df = pd.DataFrame(data=results_dict)

    fig, axes = plt.subplots()
    sns.violinplot(data=df, ax=axes, scale="width")
    # axes = df.boxplot()
    axes.set_title("Larger down payment vs Investing in S&P 500")
    axes.set_xlabel("Down payment (700k - x is invested)")
    axes.set_ylabel("15 year profit minus loan interest")
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%d"))
    axes.yaxis.grid(True)

    plt.tight_layout()
    plt.show()
