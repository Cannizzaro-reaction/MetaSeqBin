# MetaGeneBin
Course project for BIO3501, SJTU, 2025 spring



:hammer:Development Status 0501: Basic kMM algo (CPU ver) finished.

Parameters in `markov_model.py`:

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

Example of `assignment_summary`:

```
# eval mode
Execution time: 42.47 seconds
Number of genome files: 10
Number of reads: 20000
K-mer size: 7

Accuracy on test data: 0.7830 (15660/20000)

Total number of reads: 20000
Number of assigned reads: 20000

Assigned reads by genome:
NC_015722: 1644 reads
NC_015656: 1823 reads
NC_015859: 2020 reads
NC_007984: 2174 reads
NC_011138: 2112 reads
NC_009767: 1975 reads
NC_008709: 2002 reads
NC_009511: 1998 reads
NC_011126: 2110 reads
NC_013943: 2142 reads
```

```
# prediction only
Execution time: 19.88 seconds
Number of genome files: 10
Number of reads: 1876
K-mer size: 7

Total number of reads: 1876
Number of assigned reads: 1876

Assigned reads by genome:
NC_015722: 254 reads
NC_015656: 54 reads
NC_015859: 214 reads
NC_007984: 300 reads
NC_011138: 260 reads
NC_009767: 342 reads
NC_008709: 101 reads
NC_009511: 78 reads
NC_011126: 135 reads
NC_013943: 138 reads
```

