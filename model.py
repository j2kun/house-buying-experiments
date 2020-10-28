from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

ExperimentResults = Dict[str, List[int]]


@dataclass
class Model:
    name: str
    '''
    A Sampler accepts as input a number of samples n,
    and returns returns a time-ordered list of n monthly percent changes of the S&P 500.

    E.g.,

    >>> model.sampler(5)
    [0.03, -0.02, 0.015, -0.07, 0.1]
    '''
    sampler: Callable[[int], List[float]]


@dataclass
class ExperimentParameters:
    # How many years to simulate, i.e., the length of the loan
    years: int

    # The amount of cash on hand to be split between
    # a down payment and investment
    on_hand_usd: int

    house_price_usd: int
    mortgage_rate: float


def compute_loan_interest(loan_amt, yearly_rate, years):
    return loan_amt * (1 + yearly_rate / 12)**(12 * years) - loan_amt


def simulate_one(sampler, years, capital):
    num_months = years * 12
    starting_capital = capital
    samples = sampler(num_months)

    for monthly_pct_change in samples:
        capital *= (1 + monthly_pct_change / 100)
        if capital <= 0:
            return starting_capital

    return capital - starting_capital


def simulate_investment(sampler, years, capital, num_runs=1000):
    results = []
    for i in range(num_runs):
        results.append(simulate_one(sampler, years, capital))
    return results


def run_experiment(model: Model,
                   params: ExperimentParameters) -> ExperimentResults:
    minimum_down_payment = int(0.2 * params.house_price_usd)
    down_payments = []
    dp = minimum_down_payment
    while dp < min(params.house_price_usd * 0.8, params.on_hand_usd):
        down_payments.append(dp)
        dp += 50000

    loan_interests = dict(
        (dp,
         compute_loan_interest(params.house_price_usd -
                               dp, params.mortgage_rate, params.years))
        for dp in down_payments)
    capital_values = dict(
        (dp, params.on_hand_usd - dp) for dp in down_payments)

    results_dict = dict()
    for dp, capital in progressBar(capital_values.items(), prefix=model.name):
        results = simulate_investment(model.sampler,
                                      capital=capital,
                                      years=params.years)
        profits_minus_loan_interest = [
            profit - loan_interests[dp] for profit in results
        ]
        results_dict["{:d}k".format(dp // 1000)] = profits_minus_loan_interest

    return results_dict


def progressBar(iterable,
                prefix='',
                suffix='',
                decimals=1,
                length=50,
                fill='█',
                printEnd="\r"):
    """
    Copied from https://stackoverflow.com/a/34325723/438830

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    Yields a generator that passes through the items from `iterable`
    with a progress bar printed as a side effect.
    """
    total = len(iterable)

    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    printProgressBar(0)
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    print()


def violin_plot(params: ExperimentParameters,
                results: ExperimentResults,
                model_name: str = ''):
    df = pd.DataFrame(data=results)

    fig, axes = plt.subplots()
    axes.set_title("Down payment vs Investing " +
                   f'({model_name})' if model_name else '')
    axes.set_xlabel(
        f"Down payment ({params.on_hand_usd / 1000}k - x is invested)")
    axes.set_ylabel(f"{params.years} year profit minus loan interest")
    axes.set_yscale("symlog", base=100)
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%d"))
    axes.yaxis.grid(True)

    sns.violinplot(data=df, ax=axes, scale="width")
    return fig


def positive_returns_plot(params: ExperimentParameters,
                          results: ExperimentResults,
                          model_name: str = ''):
    '''
    A bar chart of the fraction of the distribution of outcomes
    that has a positive return.
    '''
    df = pd.DataFrame(data=results)
    ratios = []
    for column in df.columns:
        col = df[column]
        ratios.append(len(col[col > 0]) / len(col))

    fig, axes = plt.subplots()
    axes.set_title("Down payment vs Investing " +
                   f'({model_name})' if model_name else '')
    axes.set_xlabel(
        f"Down payment ({params.on_hand_usd / 1000}k - x is invested)")
    axes.set_ylabel(
        f"Fraction of experiments with {params.years} year positive profit")
    axes.yaxis.grid(True)

    x_pos = [i for i, _ in enumerate(df.columns)]
    axes.bar(x_pos, ratios, tick_label=list(df.columns))
    return fig
