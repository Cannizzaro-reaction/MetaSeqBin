import os
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import argparse
import csv

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
        for i in range(len(sequence) - self.k): # 0 ~ l-k-1
            kmer = sequence[i:i+self.k] # O_j
            next_kmer = sequence[i+1:i+1+self.k] # O_j+1
            
            self.kmer_counts[genome_id][kmer] += 1
            self.transition_counts[genome_id][kmer][next_kmer] += 1
    
    def train(self, genome_files):
        self.genome_files = genome_files
        for i, genome_file in enumerate(genome_files):
            # Store the filename without extension as genome name
            base_name = os.path.basename(genome_file)
            file_name = os.path.splitext(base_name)[0]
            self.genome_names.append(file_name)
            
            for record in SeqIO.parse(genome_file, "fasta"):
                sequence = str(record.seq).upper()
                self.count_kmers_and_transitions(sequence, i)
    
    def calculate_score(self, query_sequence, genome_id):
        score = 0
        
        for j in range(len(query_sequence) - self.k): # 0 ~ l-k-1
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
        # get best score and best id for classification
        best_score = float('inf')
        best_genome_id = -1
        
        for genome_id in range(len(self.genome_names)):
            score = self.calculate_score(query_sequence, genome_id)
            if score < best_score:
                best_score = score
                best_genome_id = genome_id
        
        return best_genome_id, best_score
    
    def get_all_scores(self, query_sequence):
        """Get scores for all genomes for a given query sequence"""
        scores = {}
        for genome_id in range(len(self.genome_names)):
            scores[genome_id] = self.calculate_score(query_sequence, genome_id)
        return scores

def process_reads(reads_file, model):
    # assign reads to a genome
    assigned_reads = defaultdict(list)
    total_reads = 0
    
    for record in SeqIO.parse(reads_file, "fasta"):
        total_reads += 1
        sequence = str(record.seq).upper()
        
        genome_id, score = model.classify_sequence(sequence)
        assigned_reads[genome_id].append(record.id)
    
    return assigned_reads, total_reads

def get_truth_class(seq_id_map_file, model):
    # Get mapping from genome name to ID
    genome_name_to_id = {name: i for i, name in enumerate(model.genome_names)}
    
    # Debug output
    print(f"Available genome names: {model.genome_names}")
    
    # load csv
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

def calculate_accuracy(test_file, model, ground_truth):
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
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description='Genomic sequence classification using Markov Models')
    parser.add_argument('--reads', type=str, help='Path to reads FASTA file')
    parser.add_argument('--genomes', type=str, help='Path to directory containing genome FASTA files')
    parser.add_argument('--test', type=str, help='Path to test FASTA file')
    parser.add_argument('--seq_id_map', type=str, help='Path to sequence ID mapping file')
    parser.add_argument('--k', type=int, default=11, help='Order of the Markov Model')
    parser.add_argument('--output', type=str, default='results.txt', help='Path to output file')
    parser.add_argument('--scores_output', type=str, default='sequence_scores.txt', help='Path to sequence scores output file')
    
    args = parser.parse_args()
    
    # Get genome files
    genome_files = [os.path.join(args.genomes, f) for f in os.listdir(args.genomes) 
                    if f.endswith('.fa') or f.endswith('.fasta') or f.endswith('.fna')]
    
    # Train model
    model = KMarkovModel(args.k)
    model.train(genome_files)
    
    # Calculate accuracy if test data is provided and output all scores
    if args.test:
        # Initialize ground truth if mapping file is provided
        ground_truth = {}
        if args.seq_id_map:
            ground_truth = get_truth_class(args.seq_id_map, model)

        # Process test sequences and get scores for all genomes
        with open(args.scores_output, 'w') as f_scores:
            # Write header: sequence_id, true_label, followed by genome names
            f_scores.write("sequence_id,true_label,")
            f_scores.write(",".join([model.genome_names[i] for i in range(len(model.genome_names))]))
            f_scores.write("\n")
            
            correct = 0
            total = 0
            
            # Create inverse mapping from genome_id to genome_name
            genome_id_to_name = {i: name for i, name in enumerate(model.genome_names)}
            
            for record in SeqIO.parse(args.test, "fasta"):
                sequence = str(record.seq).upper()
                scores = model.get_all_scores(sequence)
                
                # Calculate best genome
                best_genome_id = min(scores, key=scores.get)
                
                # Get true label (genome name) if available
                true_label = "unknown"
                if record.id in ground_truth:
                    total += 1
                    true_genome_id = ground_truth[record.id]
                    true_label = genome_id_to_name.get(true_genome_id, f"unknown-{true_genome_id}")
                    
                    if best_genome_id == true_genome_id:
                        correct += 1
                
                # Write sequence_id, true_label, followed by scores for all genomes
                f_scores.write(f"{record.id},{true_label}")
                for genome_id in range(len(model.genome_names)):
                    f_scores.write(f",{scores[genome_id]:.4f}")
                f_scores.write("\n")
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
    
    # Process reads if provided
    assigned_reads = defaultdict(list)
    total_reads = 0
    
    if args.reads:
        assigned_reads, total_reads = process_reads(args.reads, model)
    
    # Write results to file
    with open(args.output, 'w') as f:
        # Output accuracy if calculated
        if args.test and total > 0:
            f.write(f"Accuracy on test data: {accuracy:.4f} ({correct}/{total})\n\n")
        
        # Output reads info if processed
        if args.reads:
            f.write(f"Total number of reads: {total_reads}\n")
            f.write(f"Number of assigned reads: {sum(len(reads) for reads in assigned_reads.values())}\n\n")
            
            f.write("Assigned reads by genome:\n")
            for genome_id in range(len(model.genome_names)):
                read_count = len(assigned_reads[genome_id]) if genome_id in assigned_reads else 0
                f.write(f"{model.genome_names[genome_id]}: {read_count} reads\n")
    
    print(f"Results written to {args.output}")
    if args.test:
        print(f"Sequence scores written to {args.scores_output}")

if __name__ == "__main__":
    main()