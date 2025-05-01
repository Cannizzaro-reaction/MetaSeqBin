import os
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import csv
import argparse
import time

class KMarkovModel:
    def __init__(self, k=5):
        self.k = k
        self.transition_counts = {}  # genome_id: F_i(O_m | O_n)
        self.kmer_counts = {}  # genome_id: kmer_sequence: F_i(O_m)
        self.genome_names = []  # names of reference genomes (without extension)
        self.genome_files = []  # full paths to genome files
        
    def count_kmers_and_transitions(self, sequence, genome_id):
        if genome_id not in self.transition_counts:
            self.transition_counts[genome_id] = defaultdict(lambda: defaultdict(int))
            self.kmer_counts[genome_id] = defaultdict(int)
        
        # count k-mers and transitions
        for i in range(len(sequence) - self.k):
            kmer = sequence[i:i+self.k]
            next_kmer = sequence[i+1:i+1+self.k]
            
            # kmer: F_i(O_m)
            self.kmer_counts[genome_id][kmer] += 1
            # transition: F_i(O_m | O_n)
            self.transition_counts[genome_id][kmer][next_kmer] += 1
    
    def train(self, genome_files):
        # get genome files
        self.genome_files = genome_files
        for i, genome_file in enumerate(genome_files):
            base_name = os.path.basename(genome_file)
            file_name = os.path.splitext(base_name)[0]
            self.genome_names.append(file_name)
            
            for record in SeqIO.parse(genome_file, "fasta"):
                sequence = str(record.seq).upper()
                # get F_i(O_m) and F_i(O_m | O_n) for each sequence
                self.count_kmers_and_transitions(sequence, i)
    
    def calculate_score(self, query_sequence, genome_id):
        score = 0
        
        for j in range(len(query_sequence) - self.k):
            kmer = query_sequence[j:j+self.k]
            next_kmer = query_sequence[j+1:j+1+self.k]
            
            # check if the k-mer exists in the genome
            if kmer in self.kmer_counts[genome_id] and kmer in self.transition_counts[genome_id]:
                if next_kmer in self.transition_counts[genome_id][kmer]:
                    # calculate P_i(O_j | O_j+1)
                    probability = self.transition_counts[genome_id][kmer][next_kmer] / self.kmer_counts[genome_id][kmer]
                    score -= np.log(probability)
                else:
                    # penalty for unseen transition
                    score += 10
            else:
                # penalty for unseen k-mer
                score += 10
        
        return score
    
    def classify_sequence(self, query_sequence):
        best_score = float('inf')
        best_genome_id = -1
        
        for genome_id in range(len(self.genome_names)):
            score = self.calculate_score(query_sequence, genome_id)
            if score < best_score:
                best_score = score
                best_genome_id = genome_id
        
        return best_genome_id, best_score
    
    def get_all_scores(self, query_sequence):
        scores = {}
        for genome_id in range(len(self.genome_names)):
            scores[genome_id] = self.calculate_score(query_sequence, genome_id)
        return scores


def train_model(genomes_dir, k=11):
    # get genome files
    genome_files = [os.path.join(genomes_dir, f) for f in os.listdir(genomes_dir) 
                    if f.endswith('.fa') or f.endswith('.fasta') or f.endswith('.fna')]
    
    # train model
    model = KMarkovModel(k)
    model.train(genome_files)
    
    return model, len(genome_files)


def predict_sequences(model, reads_file, output_csv, scores_output=None):
    # dictionary to store assigned reads by genome ID
    assigned_reads = defaultdict(list)
    
    # Count number of reads
    num_reads = 0
    
    # CSV to write classification
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sequence_id', 'genome_name'])
        
        # record detailed score for assigning a read to each class
        scores_file = None
        if scores_output:
            scores_file = open(scores_output, 'w')
            scores_file.write("sequence_id,")
            scores_file.write(",".join([model.genome_names[i] for i in range(len(model.genome_names))]))
            scores_file.write("\n")
        
        # process each sequence in the reads file
        for record in SeqIO.parse(reads_file, "fasta"):
            num_reads += 1
            sequence = str(record.seq).upper()

            if scores_file:
                scores = model.get_all_scores(sequence)
                scores_file.write(f"{record.id}")
                for genome_id in range(len(model.genome_names)):
                    scores_file.write(f",{scores[genome_id]:.4f}")
                scores_file.write("\n")
                
                # get best score
                best_genome_id = min(scores, key=scores.get)
            else:
                best_genome_id, _ = model.classify_sequence(sequence)

            assigned_reads[best_genome_id].append(record.id)
            writer.writerow([record.id, model.genome_names[best_genome_id]])

        if scores_file:
            scores_file.close()
    
    return assigned_reads, num_reads


def evaluate_accuracy(model, test_file, seq_id_map_file):
    # get ground truth mappings
    ground_truth = get_truth_class(seq_id_map_file, model)
    
    # evaluate accuracy
    correct = 0
    total = 0
    
    for record in SeqIO.parse(test_file, "fasta"):
        if record.id in ground_truth:
            total += 1
            sequence = str(record.seq).upper()
            genome_id, _ = model.classify_sequence(sequence)
            if genome_id == ground_truth[record.id]:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy on test data: {accuracy:.4f} ({correct}/{total})\n")
    
    return accuracy, correct, total


def get_truth_class(seq_id_map_file, model):
    # get mapping from genome name to ID
    genome_name_to_id = {name: i for i, name in enumerate(model.genome_names)}
    
    # load ground truth
    ground_truth = {}
    try:
        with open(seq_id_map_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    seq_id = row[0]
                    genome_name = row[1]

                    if genome_name in genome_name_to_id:
                        ground_truth[seq_id] = genome_name_to_id[genome_name]
                    else:
                        print(f"Warning: No genome found with name '{genome_name}'")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Falling back to simple space-separated format")

        with open(seq_id_map_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    seq_id = parts[0]
                    try:
                        genome_id = int(parts[1])
                        ground_truth[seq_id] = genome_id
                    except ValueError:
                        if parts[1] in genome_name_to_id:
                            ground_truth[seq_id] = genome_name_to_id[parts[1]]
                        else:
                            print(f"Warning: Could not interpret '{parts[1]}' as genome ID or filename")
    
    return ground_truth


def save_assignment_summary(model, assigned_reads, output_file, accuracy_info=None, timing_info=None):
    total_reads = sum(len(reads) for reads in assigned_reads.values())
    
    with open(output_file, 'w') as f:
        # Write timing information if provided
        if timing_info:
            elapsed_time, num_genomes, num_reads, k_value = timing_info
            f.write(f"Execution time: {elapsed_time:.2f} seconds\n")
            f.write(f"Number of genome files: {num_genomes}\n")
            f.write(f"Number of reads: {num_reads}\n")
            f.write(f"K-mer size: {k_value}\n\n")
        
        # Write accuracy information if provided
        if accuracy_info:
            accuracy, correct, total = accuracy_info
            f.write(f"Accuracy on test data: {accuracy:.4f} ({correct}/{total})\n\n")
        
        f.write(f"Total number of reads: {total_reads}\n")
        f.write(f"Number of assigned reads: {total_reads}\n\n")
        
        f.write("Assigned reads by genome:\n")
        for genome_id in range(len(model.genome_names)):
            read_count = len(assigned_reads[genome_id]) if genome_id in assigned_reads else 0
            f.write(f"{model.genome_names[genome_id]}: {read_count} reads\n")


##############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genomic sequence classification using Markov Models')
    parser.add_argument('--genomes', type=str, required=True, help='Path to directory containing genome FASTA files')
    parser.add_argument('--reads', type=str, required=True, help='Reads in FASTA file that need to be assigned')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file with classifications')
    parser.add_argument('--scores_output', type=str, help='Path to sequence scores output file')
    parser.add_argument('--k', type=int, default=5, help='Order of the Markov Model')
    parser.add_argument('--eval', action='store_true', help='Run in test mode with accuracy evaluation')
    parser.add_argument('--seq_id_map', type=str, help='Path to sequence ID mapping file (required for test mode)')
    parser.add_argument('--assignment_summary', type=str, help='Path to read assignment summary output file')
    parser.add_argument('--time', action='store_true', help='Track and report execution time')
    
    args = parser.parse_args()

    start_time = time.time() if args.time else None
    
    # train model
    model, num_genomes = train_model(args.genomes, args.k)
    
    # prediction
    assigned_reads, num_reads = predict_sequences(model, args.reads, args.output_csv, args.scores_output)

    end_time = time.time() if args.time else None
    elapsed_time = end_time - start_time if args.time else None
    
    # record timing info
    timing_info = None
    if args.time:
        timing_info = (elapsed_time, num_genomes, num_reads, args.k)
        print(f"Execution time: {elapsed_time:.2f} seconds")
        print(f"Processed {num_genomes} genome files and {num_reads} reads with k={args.k}")

    # record accuracy info
    accuracy_info = None
    if args.eval:
        if not args.seq_id_map:
            print("Error: --seq_id_map is required in test mode")
        else:
            # evaluate and store accuracy information
            accuracy_info = evaluate_accuracy(model, args.reads, args.seq_id_map)

    # get summary info
    if args.assignment_summary:
        save_assignment_summary(model, assigned_reads, args.assignment_summary, accuracy_info, timing_info)