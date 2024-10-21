# Notes on Feature Engineering
From talking with Claude on my stock up/down classifier network.

### Input
- neural networks only see data you put in the input layer
- each neuron of input layer == 1 feature of your dataset, 1 x value

### Feature Engineering
- creating data from the raw data to help the network learn, trying to capture more relevant information
- for time series data, often encodes historical context
- relative changes often more useful than absolute
- represent historical data as separate features rather than a single feature sequence
- NN can infer patterns but being explicit with your feature engineering makes learning easier and more efficient
- inferring patterns requires bigger networks and more data
- feature engineering is very important - allows you to incorporate domain knowledge

### Types of Features (for finance & time series)

- Lagged features: Past values of key metrics.
- Rolling statistics: Moving averages, standard deviations over windows.
- Percentage changes: Relative changes over different periods.
- Technical indicators: RSI, MACD, etc.
- Trend features: Encoding directional movements.

### Types of Neural Networks
- Feedforward Neural Networks (FNN): Good for tabular data and engineered features.
- Convolutional Neural Networks (CNN): Excel with grid-like data.
- Recurrent Neural Networks (RNN), including LSTM: Designed for sequential data.
	- LSTM: good at capturing long-term dependencies in time series data.
- Transformer models: Good with sequential data and long-range dependencies.
