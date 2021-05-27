
# Rotowire-RG-metric
Pytorch Code for the RG metric of [Challenges in Data-to-Document Generation][1] 
(Wiseman, Shieber, Rush; EMNLP 2017)

The RG metric is computed using a neural information extraction network, which 
reads generated outputs and creates a (very simple) knowledge base of asserted facts. 
For instance the sentence _"Lebron James scored 25 points [...]"_ would yield one record [Lebron James | 25 | PTS].

These facts are then compared to true data and used as evaluation metrics for the generated outputs:

 - RG-# reports the total number of correctly asserted facts
 - RG-% reports the proportion of correct facts among all asserted facts
 
Note that you can find interesting discussion regarding Data-to-Text evaluation
(esp. on RotoWire) in the following paper:
[A Gold Standard Methodology for Evaluating Accuracy in Data-To-Text Systems][2] (Thomson, Reiter; INLG 2020)

[1]: https://arxiv.org/abs/1707.08052
[2]: https://www.aclweb.org/anthology/2020.inlg-1.6.pdf


## Why this repo?

Original code ([here](https://github.com/harvardnlp/data2text)) is written in 
Python2 and neural networks are instantiated and trained using Lua code. 
This makes it highly unusable on modern shared computing machines (due to 
incompatibly in Python/CUDA versions) and difficult to read and understand when 
trying to adapt them to other use cases.

We ported the code with a minimal-changes policy, so that behavior stays the same at all points.

Also note that we included the following list of changes, which have zero impact 
on results but provide quality of life improvements:
 
 - summaries can be considered as lists of spans (traditionally sentences), 
   which can be provided by users. By default, the script uses nltk's sent_tokenize 
   to split the summaries in sentences, but we have experimented splitting summaries
   at the clause level (e.g. _"The Hawks scored 105 points, while the Bulls scored 98 points."_
   can be split at _", while"_).
 - I have added wherever possible logging information and tqdm's progressbar


__RESULTS:__ models trained using this repo obtain the same level of recall/accuracy
than orginal models in LUA (i.e. ~95% accuracy and ~70% recall)
 
 
### Data

We assume data is in the original Rotowire format, found at `$ROTOWIRE/json`, 
where `$ROTOWIRE` is the directory in which everything is stored / ran. 
 

### Training the RG information extractor

Training data can be created using the following command:

```bash
python data_utils.py \
       -mode make_ie_data \
       -test \
       -input_path $ROTOWIRE/json \
       -output_fi $ROTOWIRE/output/training-data.h5
```

Very important step: once training data is generated, please check the file 
`$ROTOWIRE/output/training-data.labels` and have a look at the index of label NONE.
It will be used in almost all commands below via `--ignore-idx`

You can now train either an LSTM or a CONV model.
RG metric is an ensemble of 3 LSTMs and 3 CONVs models.

Note that checkpoints will be save after each epoch, with their filename indicating
the accuracy and recall on a validation set (not seen during training). I have trained
3 models of each, using arbitrary random seeds. If you want to know everything,
I've used `--seed` in `1234`, `5678`, `3435`. Each time, I selected the best
performing checkpoint in terms of accuracy & recall.

After training a model, select the checkpoint you want to keep,
rename it something like `lstm1`, and remove other checkpoints.
You can then train another model, and repeat this process.
(This is not mendatory, just easier to track which models you want to use later on)

You can train an LSTM model with 

```bash
python run.py \
       --datafile $ROTOWIRE/output/training-data.h5 \
       --save-directory $ROTOWIRE/models \
       --model lstm \
       --gpu 0 \
       --num-epochs 10 \
       --batch-size 32 \
       --embedding-size 200 \
       --hidden-dim 700 \
       --lr 1 \
       --dropout 0.5 \
       --ignore-idx 15
```

and a CONVolutional model with:

```bash
python run.py 
       --datafile $ROTOWIRE/output/training-data.h5 \
       --save-directory models \
       --model conv \
       --gpu 0 \
       --num-epochs 10 \
       --batch-size 32 \
       --embedding-size 200 \
       --hidden-dim 500 \
       --num-filters 200 \
       --lr 0.7 \
       --dropout 0.5 \
       --ignore-idx 15 
```

Warning: sometimes, training is unstable and models might learn to predict NONE
labels everytime after some epochs. Monitor your training, and restart with another
seed if this happens.


### Using RG information extractor as a metric

This steps assumes that:
 - the generated texts can be found at `$ROTOWIRE/gens/predictions.txt`
 - you want to use all models contained in `$ROTOWIRE/models/`

You first need to prep data for the RG metric, using (not that if you include `-test` 
the script will assume you want to compare to test data, while not including it means
using validation data):

```bash
python data_utils.py \
      -mode prep_gen_data \
      -test \
      -gen_fi $ROTOWIRE/gens/predictions.txt \
      -dict_pfx $ROTOWIRE/output/training-data \
      -output_fi $ROTOWIRE/output/prep_predictions.h5 \
      -input_path $ROTOWIRE/json
```

After this is done, you can compute RG scores and generate the list of extracted records,
using the following command:

```bash
python run.py \
       --just-eval \
       --datafile $ROTOWIRE/output/training-data.h5 \
       --preddata $ROTOWIRE/output/prep_predictions.h5 \
       --eval-models $ROTOWIRE/models \
       --gpu 0 \
       --test \
       --ignore-idx 15 \
       --vocab-prefix $ROTOWIRE/output/training-data 
```

Also note that if you are interested in reading the tuples created, you can use
`--show-correctness` to add a `|RIGHT` or `|WRONG` tag to each tuple, depending
on whether the generated tuple is correct or not.

[comment]: <> (### Example use of the data_utils on the sliced rotowire data &#40;in valid mode against the ref&#41;)

[comment]: <> (`ROTODIR='data/slice_based_rotowire'`)

[comment]: <> (`python data_utils.py -mode make_ie_data -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5`)

[comment]: <> (`python data_utils.py -mode prep_gen_data -gen_fi $ROTODIR/refs/ht_text_validation_BASE.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_valid_out.h5 -input_path $ROTODIR/json`)

[comment]: <> (### Example use of the data_utils on the sliced rotowire data &#40;in test mode with Puduppully AI gens&#41;)

[comment]: <> (`ROTODIR='data/orig_rotowire'`)

[comment]: <> (`python data_utils.py -mode make_ie_data -test -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5`)

[comment]: <> (`python data_utils.py -mode prep_gen_data -test -gen_fi $ROTODIR/gens/rebuffel_test_gen_not_paper.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_test_out.h5 -input_path $ROTODIR/json`)

