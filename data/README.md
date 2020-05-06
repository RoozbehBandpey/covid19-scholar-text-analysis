Download the CORD-19 data from [here](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) after unpacking place it in this directory


The data consist of 57k scholarly article about COVID-19

There so much new information that is being pushed fast, and that makes it super difficult for the researchers to get insight over the true behavior of the virus. There are so much variability about many aspect of the virus that makes us think about it mysteriously. It turns out NLP and big data technologies can provide a great help in this are.

The data consists of multipe subdirectories and metadata
in CORD-19-research-challenge\biorxiv_medrxiv\biorxiv_medrxiv\pdf_json
We have 2278 item in json format

* \CORD-19-research-challenge\comm_use_subset\comm_use_subset\pdf_json 9769
* \CORD-19-research-challenge\comm_use_subset\comm_use_subset\pmc_json 9390
* \CORD-19-research-challenge\custom_license\custom_license\pdf_json 31376
* \CORD-19-research-challenge\custom_license\custom_license\pmc_json 10615
* \CORD-19-research-challenge\noncomm_use_subset\noncomm_use_subset\pdf_json 2518
* \CORD-19-research-challenge\noncomm_use_subset\noncomm_use_subset\pmc_json 2258


Articles are in json format which is great, 

Run load to get basic insight over data
TODO: get more insight over data structure with regex
* Create schema loader based on data structure with sqlalchemy



The loader.py parses the data and loads it into sql database for further training and usage


In an era of transfer learning, transformers, and deep architectures, we believe that pretrained models provide a unified solution to many real-world problems and allow handling different tasks and languages easily.

