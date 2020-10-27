from model import ExperimentParameters
from model import Model
from model import plot
from model import run_experiment
from scipy.stats import gaussian_kde


def fit_kde(data):
    '''
    A model which assumes each monthly change is IID.
    So we fit a Kernel Density Estimator to the data
    and sample as needed.
    '''
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

    plot(params, run_experiment(model, params))
