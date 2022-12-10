#!/usr/bin/env bash

function fail {
    printf '%s\n' "$1" >&2  ## Send message to stderr. Exclude >&2 if you don't want it that way.
    exit "${2-1}"  ## Return a code specified by $2 or 1 by default.
}

if ! command -v 7z &> /dev/null
then
    echo "This script requires 7z to extract the data file"
    echo "On a Debian-like system can be done with:"
    echo "$ sudo apt-get install p7zip-full"
    exit 1
fi

wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget \
  --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
  'https://docs.google.com/uc?export=download&id=1IdYfoF3oZ_ZiRmuQTrgUIiSnn3DppxEu' -O- | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IdYfoF3oZ_ZiRmuQTrgUIiSnn3DppxEu" \
  -O data.7z && rm -rf /tmp/cookies.txt || fail "can't download"
7z x data.7z || fail "can't extract"
rm data.7z || fail "can't remove download"