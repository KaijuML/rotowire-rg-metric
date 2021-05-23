# ie-extractor
Code for the RG metric of Challenges in Data-to-Document Generation (Wiseman, Shieber, Rush; EMNLP 2017)

# Example use of the data_utils on the sliced rotowire data (in valid mode against the ref)
ROTODIR='data/slice_based_rotowire'
echo $ROTODIR/output/output.h5
python data_utils.py -mode make_ie_data -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5
python data_utils.py -mode prep_gen_data -gen_fi $ROTODIR/refs/ht_text_validation_BASE.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_valid_out.h5 -input_path $ROTODIR/json

ROTODIR='data/orig_rotowire'
# Example use of the data_utils on the sliced rotowire data (in test mode with Puduppully AI gens)
python data_utils.py -mode make_ie_data -test -input_path $ROTODIR/json -output_fi $ROTODIR/output/output.h5
python data_utils.py -mode prep_gen_data -test -gen_fi $ROTODIR/gens/rebuffel_test_gen_not_paper.txt -dict_pfx $ROTODIR/output/output -output_fi $ROTODIR/output/prep_gen_test_out.h5 -input_path $ROTODIR/json
