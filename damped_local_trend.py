'''
A model that uses https://github.com/uber/orbit
'''
from orbit.diagnostics.plot import plot_predicted_data
from orbit.models import dlt
import numpy as np
import pandas as pd

df = pd.read_csv("snp.tsv", sep='\t', parse_dates=['Month'])[:-12*80]
df = df.sort_values(by='Month')
test_size = 12*5
train_df = df[:-test_size]
test_df = df[-test_size:]

model = dlt.DLTMAP(
    response_col='Percent change',
    date_col='Month',
)
model.fit(df=train_df)

# predicted df
predicted_df = model.predict(df=test_df)
print(predicted_df)

# plot predictions
plot_predicted_data(training_actual_df=train_df,
                    predicted_df=predicted_df,
                    date_col=model.date_col,
                    actual_col=model.response_col,
                    test_actual_df=test_df)
