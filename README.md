
# Rotowire-RG-metric
Pytorch Code for the RG metric of [Challenges in Data-to-Document Generation][1] 
(Wiseman, Shieber, Rush; EMNLP 2017)

The RG metric is computed using a neural information extraction network, which 
reads generated outputs and creates a (very simple) knowledge base of asserted facts. 
For instance the sentence _"Lebron James scored 25 points [...]"_ would yield one record [Lebron James | 25 | PTS].

This facts are then compared to true data and used as evaluation metrics for the generated outputs:

 - RG-# reports the total number of correctly asserted facts
 - RG-% reports the proportion of correct facts among all asserted facts
 
Note that this method has several issues, which are talked about in 
[A Gold Standard Methodology for Evaluating Accuracy in Data-To-Text Systems][2] (Thomson, Reiter; INLG 2020)

[1]: https://arxiv.org/abs/1707.08052
[2]: https://www.aclweb.org/anthology/2020.inlg-1.6.pdf


## Why this repo?

Original code can be found [here](https://github.com/harvardnlp/data2text). 

Orinal utils files are written in Python2 and neural networks are instanciated
and trained using Lua code. This makes it highly unusable on modern shared computing 
machines (due to incopatibily in Python/CUDA versions) and difficult to read and
understand when trying to adapt them to other use cases.

We ported the code with a minimal change policies, so that behaviors stays the same at all points.

Also note that we included the following list of change, which have zero impact 
on results but provide quality of life improvements:
 
 - summaries can be considered as lists of spans (tradionality sentences), 
   which can be provided by users. By default, the script uses nltk's sent_tokenize 
   to split the summaries in sentences, but we have experimented spliting summaries
   at the clause level (e.g. _"The Hawks scored 105 points, while the Bulls scored 98 points."_
   can be split at _", while"_).
 - I have added wherever possible logging information and tqdm's progressbar
 
 
### Data

We assume data is in the original Rotowire format, found at `$ROTOWIRE/json`, 
where `$ROTOWIRE` is the directory in which everything is stored / ran. 
 

### Training the RG information extractor

Training data can be created using the following command:

```bash
python data_utils.py -mode make_ie_data -test -input_path $ROTOWIRE/json -output_fi $ROTOWIRE/output/training-data.h5
```

You can train an LSTM model with 

```bash
python run.py --datafile $ROTOWIRE/output/training-data.h5 --save-directory $ROTOWIRE/models --gpu 0 --num-epochs 10 --model lstm --embedding-size 200 --hidden-dim 700 --dropout 0.3 --ignore-idx 15
```

and a CONVolutional model with:

```bash
python run.py --datafile $ROTOWIRE/output/training-data.h5 --save-directory $ROTOWIRE/models --gpu 0 --epochs 10 --model lstm --embedding-size 200 --hidden-dim 700 --dropout 0.3
```

Please note that a checkpoint will be saved every epoch, with validation accuracy and recall in its filename.
Select the one you want to keep, rename it something like `lstm1`, and remove other checkpoints.
You can then train another model, and repeat this process.

### Using RG information extractor as a metric

Assuming the generated texts can be found at `$ROTOWIRE/gens/predictions.txt`,
you can compute its RG score using the following command:

```bash

```

### Example use of the data_utils on the sliced rotowire data (in valid mode against the ref)
`ROTODIR='data/slice_based_rotowire'`

`python data_utils.py -mode make_ie_data -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5`

`python data_utils.py -mode prep_gen_data -gen_fi $ROTODIR/refs/ht_text_validation_BASE.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_valid_out.h5 -input_path $ROTODIR/json`

### Example use of the data_utils on the sliced rotowire data (in test mode with Puduppully AI gens)
`ROTODIR='data/orig_rotowire'`

`python data_utils.py -mode make_ie_data -test -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5`

`python data_utils.py -mode prep_gen_data -test -gen_fi $ROTODIR/gens/rebuffel_test_gen_not_paper.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_test_out.h5 -input_path $ROTODIR/json`

