from kernel_density import fit_kde
from markov_chain import fit_markov_chain
from model import ExperimentParameters
from model import Model
from model import positive_returns_plot
from model import run_experiment
from model import violin_plot
from scipy.stats import gaussian_kde
import pandas as pd


def model_filename(model_name):
    return model_name.lower().replace(' ', '-')


fitters = [fit_kde, fit_markov_chain]
data = pd.read_csv("snp.tsv", sep='\t')

for fitter in fitters:
    model = fitter(data.head(50 * 12))

    params = ExperimentParameters(
        years=30,
        house_price_usd=1000000,
        on_hand_usd=700000,
        mortgage_rate=0.03,
    )

    experiment_results = run_experiment(model, params)

    results_fig = violin_plot(params, experiment_results, model.name)
    results_fig.savefig(f'{model_filename(model.name)}-violin.svg',
                        bbox_inches='tight')

    returns_fig = positive_returns_plot(params, experiment_results, model.name)
    returns_fig.savefig(f'{model_filename(model.name)}-returns.svg',
                        bbox_inches='tight')
