'''
A model that uses Facebook's Prophet
'''

from fbprophet import Prophet
from model import ExperimentParameters
from model import Model
from model import run_experiment
from model import violin_plot
import matplotlib.pyplot as plt
import pandas as pd


class ProphetSampler:
    def __init__(self, model):
        self.model = model
        self.samples = None
        self.column = 0

    def sample(self, n):
        if self.samples is None or self.column >= len(self.samples[0]):
            print("resampling...")
            future = self.model.make_future_dataframe(periods=n,
                                                      freq='M',
                                                      include_history=False)
            self.samples = self.model.predictive_samples(future)['yhat']
            self.column = 0

        output = [x[self.column] for x in self.samples]
        self.column += 1
        return output


def fit_prophet(df):
    df = df.sort_values(by='Month')
    df = df[['Month', 'Percent change']]
    df = df.rename(columns={'Month': 'ds', 'Percent change': 'y'})

    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    sampler = ProphetSampler(m)

    return Model(name='Prophet', sampler=sampler.sample)


if __name__ == "__main__":
    data = pd.read_csv("snp.tsv", sep='\t', parse_dates=['Month'])
    model = fit_prophet(data.head(50 * 12))

    params = ExperimentParameters(
        years=30,
        house_price_usd=1000000,
        on_hand_usd=700000,
        mortgage_rate=0.03,
    )

    violin_plot(params, run_experiment(model, params), model.name)
    plt.show()
