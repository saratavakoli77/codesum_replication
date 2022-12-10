#!/bin/bash
# adapted from https://github.com/EdinburghNLP/code-docstring-corpus/blob/
#   60d0883f8ff8a928a17156799251aa6cd12999df/scripts/nmt/prepare_data_declbodies2desc.sh

function fail {
    printf '%s\n' "$1" >&2  ## Send message to stderr. Exclude >&2 if you don't want it that way.
    exit "${2-1}"  ## Return a code specified by $2 or 1 by default.
}

MOSES=~/mosesdecoder
function setupMoses {
  git clone https://github.com/moses-smt/mosesdecoder.git $MOSES || fail "bad moses"
  cd $MOSES || fail "no moses"
  # Checkout specific commit for consistency. They don't give what version they
  # used in the dataset, so just going to chose one.
  git checkout 4c5e89f075a72c1e45db8ba0bf5aeb38b95950ee || fail "no checkout"
  cd - || fail "?"
}
[ ! -d $MOSES ] && setupMoses

cd ../data/other || fail "no data?"

# Clone the dataset repo if it doesn't exist
[ ! -d "code-docstring-corpus" ] && git clone https://github.com/EdinburghNLP/code-docstring-corpus.git

cd code-docstring-corpus/parallel-corpus || fail "no corpus dir?"

# Insert paths to tools
# Make sure moses exists
pip install subword-nmt==0.3.7
#NEMATUS=/path/to/nematus

gunzip ./*.train.gz

cat data_ps.declbodies.test | iconv -c --from UTF-8 --to UTF-8 | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.test.tok.db || fail "f1"
cat data_ps.declbodies.valid | iconv -c --from UTF-8 --to UTF-8 | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.valid.tok.db || fail "f2"
cat data_ps.declbodies.train | iconv -c --from UTF-8 --to UTF-8 | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.train.tok.db || fail "f3"

cat data_ps.descriptions.test | iconv -c --from UTF-8 --to UTF-8  | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.test.tok.d || fail "f3"
cat data_ps.descriptions.valid | iconv -c --from UTF-8 --to UTF-8 | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.valid.tok.d || fail "f4"
cat data_ps.descriptions.train | iconv -c --from UTF-8 --to UTF-8 | $MOSES/scripts/tokenizer/tokenizer.perl > data_ps.declbodies2desc.train.tok.d || fail "f5"

$MOSES/scripts/training/clean-corpus-n.perl data_ps.declbodies2desc.train.tok db d data_ps.declbodies2desc.train.tok.clean 2 400 || fail "f6"

cat data_ps.declbodies2desc.train.tok.clean.db data_ps.declbodies2desc.train.tok.clean.d > data_ps.declbodies2desc.train.tok.clean.merged || fail "f7"
subword-nmt learn-bpe -s 89500 < data_ps.declbodies2desc.train.tok.clean.merged > data_ps.declbodies2desc.digram.model || fail "f8"

subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.train.tok.clean.db > data_ps.declbodies2desc.train.bpe.clean.db || fail "f9"
subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.train.tok.clean.d > data_ps.declbodies2desc.train.bpe.clean.d || fail "f10"
subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.valid.tok.db > data_ps.declbodies2desc.valid.bpe.db || fail "f11"
subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.valid.tok.d > data_ps.declbodies2desc.valid.bpe.d || fail "f12"
subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.test.tok.db > data_ps.declbodies2desc.test.bpe.db || fail "f13"
# In origional version (see top link) they don't
# seem to make a test.bpe.d ? This is really wierd. Not sure how they are evaluating
# things. We are going to actually make one though.
subword-nmt apply-bpe -c data_ps.declbodies2desc.digram.model < data_ps.declbodies2desc.test.tok.d > data_ps.declbodies2desc.test.bpe.d || fail "f14"

cat data_ps.declbodies2desc.train.bpe.clean.db data_ps.declbodies2desc.train.bpe.clean.d > data_ps.declbodies2desc.train.bpe.clean.merged || fail "f15"
#$NEMATUS/data/build_dictionary.py data_ps.declbodies2desc.train.bpe.clean.mergedh || fail "f15"