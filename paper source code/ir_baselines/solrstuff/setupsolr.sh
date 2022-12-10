#!/usr/bin/env bash
# For legacy reasons the container name is just 'toy_solr0'
docker run --name toy_solr0 -d -p 8983:8983 --env SOLR_HEAP=2g -t solr:8.3.1
