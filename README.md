# Using BERT on Azure Machine Learning to explore COVID-19 data

In the current global pandemic caused by COVID-19 virus, leveraging any technological tools for analyzing current data is helpful for scientist to prepare vaccine as soon as possible.

In this repository [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset is used to gain insight over hidden behavior of the virus.
[Microsoft Machine Learning](https://docs.microsoft.com/en-us/azure/synapse-analytics/sql-data-warehouse/) is used for training platform

## Data
CORD-19 is a huge body of scientific papers over COVID-19 virus
is located in ./data/CORD-19-research-challenge folder but ignored due to massive size, download data from the link above and place it in the directory and then run the script to load it
Data can reside in data lake as well as your local machine, change the target for loading to sql pool
Read more on [README](data/README.md)
## Tools and Requirements

### Why Azure Machine Learning?
