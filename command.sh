# evaluate the result and get accuracy
## optional parameters: --scores_output & --assignment_summary & --time
python markov_model.py \
    --reads data/test.fa \
    --genomes data/genomes \
    --output_csv classification_result.csv \
    --scores_output score_detail.csv \
    --k 7 \
    --eval \
    --seq_id_map data/seq_id.csv \
    --assignment_summary assignment_summary.txt \
    --time

##merge mode
python markov_model.py \
	--combine\
    --reads data/test.fa \
    --genomes data/genomes \
    --output_csv classification_result.csv \
    --scores_output score_detail.csv \
    --eval \
    --seq_id_map data/seq_id.csv \
    --assignment_summary assignment_summary.txt \
    --k_values 3 4 5 \
    --weights 0.609 0.639 0.664\
    --time

# only assign the reads (without evaluation)
## optional parameters: --scores_output & --assignment_summary & --time
python markov_model.py \
    --reads data/reads.fa \
    --genomes data/genomes \
    --output_csv classification_result_noEval.csv \
    --scores_output score_detail_noEval.csv \
    --k 7 \
    --assignment_summary assignment_summary_noEval.txt \
    --time
