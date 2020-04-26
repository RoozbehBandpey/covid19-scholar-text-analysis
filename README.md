# Using Azure Synapse to explore COVID-19 data

In the current global pandemic caused by COVID-19 virus, leveraging any technological tools for analyzing current data is helpful for scientist to prepare vaccine as soon as possible.

In this repository [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset is used to gain insight over hidden behavior of the virus.
The analytic tool that is used is [Microsoft Azure Synapse](https://docs.microsoft.com/en-us/azure/synapse-analytics/sql-data-warehouse/)

## Data
CORD-19 is a huge body of scientific papers over COVID-19 virus
is located in ./data/CORD-19-research-challenge folder but ignored due to massive size, download data from the link above and place it in the directory and then run the script to load it
Read more on [README](data/README.md)
## Tools and Requirements

### Why Azure Synapse?
Azure Synapse is an analytics service that brings together enterprise data warehousing and Big Data analytics. Although using Synapse for purpose of exploring CORD-19 dataset might sound like an overkill because Synapse is initially designed to handel data and concurrent queries on way more heavier workloads, but it has also significant capabilities to ingest, prepare, manage, and serve data for immediate BI and machine learning needs. Since Azure Synapse is new in market, the main purpose of using it is experimental.

[Real power of Azure Synapse](https://www.youtube.com/watch?v=xzxjpQSvDEA)



SQL pool represents a collection of analytic resources that are being provisioned when using Synapse SQL. The size of SQL pool is determined by Data Warehousing Units (DWU).

When the data is ready for complex analysis, Synapse SQL pool uses PolyBase to query the big data stores. PolyBase uses standard T-SQL queries to bring the data into Synapse SQL pool tables.