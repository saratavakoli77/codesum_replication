# Code to Comment “Translation”: Data, Metrics, Baselining & Evaluation

# Installation

The project has requirements of python, R, and docker.

It is recommended you use either conda or virtualenv for managing the python 
environment. Conda can be installed with the setup script found at
[online](https://docs.conda.io/en/latest/miniconda.html).

Once conda is on the machine, we can configure the environment and install the
requirements.
```bash
conda create --name codecom python=3.7
conda activate codecom
pip install -r requirements.txt
# We use the NeuralCodeSum package to match their eval funcs. It's packaged
# in here but needs to be installed.
pip install -e otherlib/NeuralCodeSum
```

# Downloading the data

A collection of all the datasets as used can be found at in the 
[https://drive.google.com/drive/u/1/folders/1yceN-raSj3M12ajHc1aGdDAU1kRcC5jP](associated google drive).

It is recommended to download and extract such that a the data folder exists
at the root of the project.

Alternatively the CODE_COM_DATASETS environment variable can be set to the extracted path.

The `getdata.sh` script will try to do this for you
 
Note this is will take several gigabytes of disk space.

# Dataset Analysis (Section 4.1)

For recreating Figures 1 and 2 please see 
//datasets_anyl/replication_package/zipf_plot.ipynb

For recreating Figure 3 please see 
//datasets_anyl/replication_package/dropoff_plots.ipynb
with additional statistical analysis in 
//datasets_anyl/replication_package/bleu-ccd.rmd

# Analyze Input-output similarity (Section 4.2)

For recreating Figures 4 please see 
//datasets_anyl/replication_package/bivariate_plots.ipynb

# IR baseline (Section 4.3)

To run the IR baselines we use [Apache Solr](https://lucene.apache.org/solr/). 
To make this setup easier, we run it in docker.

## Setting up docker
First make sure you [docker installed](https://docs.docker.com/get-docker/) on 
your machine. Then create a container for Solr:
```bash
# For legacy reasons the container name is just 'toy_solr0'
docker run --name toy_solr0 -d -p 8983:8983 --env SOLR_HEAP=2g -t solr:8.3.1
```

## Running IR Baseline

We get data into Solr by first exporting the tokenized training split into
a csv file. We can use the following script to do that.
```bash
python3 -m ir_baselines.csv_for_solr --dataset <dataset>
# Use help to see the dataset options
python3 -m ir_baselines.csv_for_solr --help
```
After we have should have a csv in data/csv_files of all the training examples
with tokenization applied (like basic CamelCaseSplitting and stop word removal). 
We can move this over to the docker container and index all the examples.
```bash
python -m ir_baselines.index_dataset_csv --dataset <dataset>
```

Finally we can run the search on all of the test set and evalutate the results
```bash
python3 -m ir_baselines.runsolr --dataset <dataset>
```

# Affinity Data Analysis (Section 4.4)

We include with the files a scrape of the comments of 1000 top Github projects.
However, if you would like to rescrape this you can use
```bash
python -m affinity_data.affinity_generator
```
Note that depending on when you run this you might get slightly different results
than the included scrape as projects evolve.

To sample the affinity data and measure the numbers in Table 3 the following
script can be used.
```bash
# The following can take about 30 minutes to run
python -m affinity_data.analyze_affinity
```
This will include numbers for the "Unfiltered" and the "Filtered" data 
(removing overridden methods, getters/setters, and very short comments).

To create the plots shown in figure 5 the following can be used:
```bash
python -m affinity_data.dist_plots
```

