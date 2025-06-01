# MetaGeneBin
Course project for BIO3501, SJTU, 2025 spring



## :hammer:Environment

Python = 3.9.19

| Package      | Version | Channel  |
| ------------ | ------- | -------- |
| numpy        | 1.26.3  | pypi     |
| biopython    | 1.78    | anaconda |
| scikit-learn | 1.5.2   | pypi     |
| pycuda       | 2025.1  | pypi     |



## :old_key:Algorithm Overview

### Single KMM Model

We use a $k^{th}$ order Markov model to classify DNA sequences , which captures the transition probabilities between k-mer oligonucleotides. Here's how it works:

* **Model Construction**:

  For each genome in the reference database, a Markov chain is built using k-mers (short DNA words of length $k$). Each state in the chain represents a k-mer, and transitions represent how often one k-mer is followed by another.

  The transition probability from k-mer $O_m$ to $O_n$ is calculated based on the following computational formula:

  $$kMM_{i,mn} = P_i(O_m | O_n) = \frac{F_i(O_m|O_n)}{F_i(O_m)}$$

  where $O_m$ and  $O_m$ are oligonucleotides of length $k$, $P(O_m|O_n)$ represents the transition probability from  $O_m$ to $O_n$,  $F(O_m|O_n)$ represents observed count of transitions from  $O_m$ to $O_n$ in a genomic sequence $i$ and $F(O_m)$ is the observed count of $O_m$.

* **Sequence Scoring**:

  For a query read, the method calculates a score $S_i$ for each reference genome $i$:

  $$S_i = -\sum_{j = 0}^{l - k - 1}ln(P_i(O_{j} | O_{j+1}))$$

  where $O_j$ and $O_{j+1}$ are two oligonucleotides of length $k$, and $P(O_{j} | O_{j+1})$ is the transition probability from $O_j$ to $O_{j+1}$ observed in the $i^{th}$ genome. When the transition from $O_j$ to $O_{j+1}$ does not exist in the $i^{th}$ genome, the logarithm value of the transition probability will be set to a constant (default is 10). 

  The genome with the lowest total score (i.e., highest likelihood) is considered the best match.



### Merged Markov Model

* Weighted score method:

  In this method, multiple k-th order Markov models (each trained with a different value of k) are used to compute scores for each genome. For a given read, each model generates a likelihood-based score for every genome. These scores are then combined using a weighted sum:

  $$Merged Score_g = \sum_{i=1}^3 w_i \cdot Score_{g}^{(k_i)}$$

  where $\text{Score}_{g}^{(k_i)}$ is the score for genome g from the i-th model with order $k_i$, $w_i$ is the normalized weight assigned to the i-th model (in our project the original weights are based on eah model's individual accuracy).

  The genome with the lowest merged score is selected as the predicted label for the read.

* Vote method:

  In this approach, each of the three models independently predicts a genome label for a read. The final prediction is based on majority voting:

  If two or more models choose the same genome, that genome is selected. If all three models predict different genomes, the prediction from the model with the largest k is chosen, based on the assumption that higher-order models capture more informative sequence context.

* Adaptive weight method:

  In this approach, we used "confidence" to calculate the weight of each of the single model we choose. The "confidence" is defined as follow:

  $$Confidence_k(S)=\frac{Score_{2nd best}(S)}{Score_{best}(S)+\epsilon}$$

  From the equation above, we can find that if there's a huge gap between the best score and the second best, which may mean that the model is quite confident about its choice, the confidence score will be higher. We then calculate the confidence score from each of the model, and normalize the weight based on the relative value of our confidence score. The normalize equation comes as follow:

  $$\omega_k^{normalized}=\frac{Confidence}{\Sigma_j Confidence_j}$$

  After that, for a certain genome $G_i$, we can calculate the weighted rank as follow:

  $$WeightedRank(G_i)=\Sigma_k\omega_k\times RankScore_k(G_i)$$

  $$RankScore_k(G_i)=N-rank_k(G_i)$$

  $N$ is the total number of genomes.



## :computer:Usage

### Basic Parameters

| Argument               | Type | Required? | Detail                                                       |
| ---------------------- | ---- | --------- | ------------------------------------------------------------ |
| `--genomes`            | str  | true      | The directory with several genome files in FASTA format. These genome sequences are used to train the model. The classification result will be given as the file name in this directory. |
| `--reads`              | str  | true      | The reads (in FASTA) format that needs to be predicted.      |
| `--output_csv`         | str  | true      | Classification result file (CSV). The first column is the ID of the reads. The second is the name of the genome which this read is assigned to. |
| `--scores_output`      | str  | false     | The file (CSV) recording detailed scores of assigning each reads to each genome. |
| `--k`                  | int  | true      | The length of sequences in kMM model.                        |
| `--eval`               | bool | false     | Whether you need to predict only or you want to calculate the accuracy of this model after prediction. If true, then you must provide a file with ground truth of each reads you're going to predict. The ground truth file should be in the same format as `output_csv`. |
| `--seq_id_map`         | str  | false     | The ground truth of the reads you're going to predict with kMM model. Must provide if you choose `--eval`. |
| `--assignment_summary` | str  | false     | A file (TXT) recording the summary of this prediction, including accuracy (in `eval` mode), running time (in `time` mode), and the number of reads assigned to each genome. |
| `--time`               | bool | false     | If true, the running time will be recorded. The result will be given along with k value, the number of reads and the number of genome sequences. |



### Running Commands

* Single model (CPU version):

  ```bash
  # evaluate the result and get accuracy
  ## optional parameters: --scores_output & --assignment_summary & --time
  python markov_model.py \
      --reads data/test.fa \
      --genomes data/genomes \
      --output_csv classification_result.csv \
      --scores_output score_detail.csv \
      --k 9 \
      --eval \
      --seq_id_map data/seq_id.csv \
      --assignment_summary assignment_summary.txt \
      --time
  
  # only assign the reads (without evaluation)
  ## optional parameters: --scores_output & --assignment_summary & --time
  ## test commands are not repeated for GPU ver. and merged models below
  python markov_model.py \
      --reads data/reads.fa \
      --genomes data/genomes \
      --output_csv classification_result_noEval.csv \
      --scores_output score_detail_noEval.csv \
      --k 7 \
      --assignment_summary assignment_summary_noEval.txt \
      --time
  ```

* Single model (GPU version):

  ```bash
  python gpu_kmm.py \
      --reads data/test.fa \
      --genomes data/genomes \
      --output_csv classification_result.csv \
      --scores_output score_detail.csv \
      --k 9 \
      --eval \
      --seq_id_map data/seq_id.csv \
      --assignment_summary assignment_summary.txt \
      --time
  ```

* Weighted score method:

  ```bash
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
  ```

* Vote method:

  ```bash
  python vote_kmm.py \
      --genomes data/genomes \
      --reads data/test.fa \
      --output_csv classification_result_k678.csv \
      --k 6 7 8 \
      --detailed_output score_detail_k678.csv \
      --eval \
      --seq_id_map data/seq_id.csv \
      --assignment_summary assignment_summary_k678.txt \
      --time
  ```

* Adaptive weight method:

  ```bash
  python adaptive_kmm.py \
      --genomes data/genomes \
      --reads data/test.fa \
      --output_csv classification_result_k678.csv \
      --k 6 7 8 \
      --detailed_output score_detail_k678.csv \
      --eval \
      --seq_id_map data/seq_id.csv \
      --assignment_summary assignment_summary_k678.txt \
      --time
  ```

  



