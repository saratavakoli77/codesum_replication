#!/usr/bin/env bash

function fail {
    printf '%s\n' "$1" >&2  ## Send message to stderr. Exclude >&2 if you don't want it that way.
    exit "${2-1}"  ## Return a code specified by $2 or 1 by default.
}

# A script for handling creating a solr core for a datset and
# indexing the data produced by the create csv scripts
if [ -z ${1+x} ]; then echo "first arg is dataset" && exit 1; else echo "dataset is set '$1'"; fi

# Delete old core (will just let it fail if doesn't exist yet)
echo "DELETE OLD CORE"
docker exec -it --user=solr toy_solr0 bin/solr delete -c "$1"
sleep 3  # Add in sleeps to try and prevent occasional crash?
# Create a core
echo "MAKE NEW CORE"
docker exec -it --user=solr toy_solr0 bin/solr create_core -c "$1"
sleep 3
# Copy in the csv
echo "COPY OVER DATA INTO DOCKER"
docker cp "$2/$1.csv" "toy_solr0:/$1-foo.csv" || fail "cant copy"
sleep 5
# Index data
echo "INDEX DATA"
docker exec -it --user=solr toy_solr0 bin/post -c "$1" "/$1-foo.csv"
sleep 3
