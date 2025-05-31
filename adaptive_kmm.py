import os
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import csv
import argparse
import time
from sklearn.metrics import precision_recall_fscore_support


class KMarkovModel:
    def __init__(self, k=5):
        self.k = k
        self.transition_counts = {}
        self.kmer_counts = {}
        self.genome_names = []
        self.genome_files = []
        
    def count_kmers_and_transitions(self, sequence, genome_id):
        if genome_id not in self.transition_counts:
            self.transition_counts[genome_id] = defaultdict(lambda: defaultdict(int))
            self.kmer_counts[genome_id] = defaultdict(int)
        
        for i in range(len(sequence) - self.k):
            kmer = sequence[i:i+self.k]
            next_kmer = sequence[i+1:i+1+self.k]
            
            self.kmer_counts[genome_id][kmer] += 1
            self.transition_counts[genome_id][kmer][next_kmer] += 1
    
    def train(self, genome_files):
        self.genome_files = genome_files
        for i, genome_file in enumerate(genome_files):
            base_name = os.path.basename(genome_file)
            file_name = os.path.splitext(base_name)[0]
            self.genome_names.append(file_name)
            
            for record in SeqIO.parse(genome_file, "fasta"):
                sequence = str(record.seq).upper()
                self.count_kmers_and_transitions(sequence, i)
    
    def calculate_score(self, query_sequence, genome_id):
        score = 0
        
        for j in range(len(query_sequence) - self.k):
            kmer = query_sequence[j:j+self.k]
            next_kmer = query_sequence[j+1:j+1+self.k]
            
            if kmer in self.kmer_counts[genome_id] and kmer in self.transition_counts[genome_id]:
                if next_kmer in self.transition_counts[genome_id][kmer]:
                    probability = self.transition_counts[genome_id][kmer][next_kmer] / self.kmer_counts[genome_id][kmer]
                    score -= np.log(probability)
                else:
                    score += 10
            else:
                score += 10
        
        return score
    
    def get_all_scores(self, query_sequence):
        scores = {}
        for genome_id in range(len(self.genome_names)):
            scores[genome_id] = self.calculate_score(query_sequence, genome_id)
        return scores
    
    def classify_sequence(self, query_sequence):
        scores = self.get_all_scores(query_sequence)
        best_genome_id = min(scores, key=scores.get)
        return best_genome_id, scores[best_genome_id]


class AdaptiveWeightedKMM:
    def __init__(self, k_values=[3, 5, 7]):
        self.k_values = k_values
        self.models = {}
        self.genome_names = []
        self.average_weights = {}
        self.weight_counts = {}
        
        # Initialize models for each k value
        for k in k_values:
            self.models[k] = KMarkovModel(k)
            self.average_weights[k] = 0.0
            self.weight_counts[k] = 0
    
    def train(self, genome_files):
        """Train all base models"""
        print(f"Training {len(self.k_values)} base models...")
        
        for k in self.k_values:
            print(f"Training k={k} model...")
            self.models[k].train(genome_files)
        
        self.genome_names = self.models[self.k_values[0]].genome_names
        print("Base models training completed.")
    
    def classify_sequence(self, sequence):
        all_scores = {}
        weights = {}
        
        # Calculate confidence-based weights for each model
        for k in self.k_values:
            scores = self.models[k].get_all_scores(sequence)
            all_scores[k] = scores
            
            # Calculate weight based on confidence (ratio of 2nd best to best score)
            sorted_scores = sorted(scores.values())
            if len(sorted_scores) >= 2:
                confidence = sorted_scores[1] / (sorted_scores[0] + 1e-8)  # Higher ratio = more confident
                weights[k] = confidence
            else:
                weights[k] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        for k in weights:
            weights[k] /= total_weight
            # Update running average of weights
            self.average_weights[k] += weights[k]
            self.weight_counts[k] += 1
        
        # Weighted voting based on ranks
        weighted_ranks = defaultdict(float)
        for k in self.k_values:
            scores = all_scores[k]
            sorted_genomes = sorted(scores.keys(), key=lambda g: scores[g])
            
            for rank, genome_id in enumerate(sorted_genomes):
                # Higher rank score for better ranking (lower score)
                rank_score = len(self.genome_names) - rank
                weighted_ranks[genome_id] += weights[k] * rank_score
        
        best_genome = max(weighted_ranks, key=weighted_ranks.get)
        return best_genome, {
            'weighted_ranks': dict(weighted_ranks),
            'weights': weights,
            'all_scores': all_scores
        }
    
    def get_normalized_weights(self):
        """Get the average normalized weights across all sequences"""
        normalized_weights = []
        for k in self.k_values:
            if self.weight_counts[k] > 0:
                avg_weight = self.average_weights[k] / self.weight_counts[k]
                normalized_weights.append(avg_weight)
            else:
                normalized_weights.append(0.0)
        return normalized_weights


def train_model(genomes_dir, k_values=[3, 5, 7]):
    genome_files = [os.path.join(genomes_dir, f) for f in os.listdir(genomes_dir) 
                    if f.endswith('.fa') or f.endswith('.fasta') or f.endswith('.fna')]
    
    model = AdaptiveWeightedKMM(k_values)
    model.train(genome_files)
    
    return model, len(genome_files)


def predict_sequences(model, reads_file, output_csv, detailed_output=None):
    assigned_reads = defaultdict(list)
    num_reads = 0
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sequence_id', 'genome_name'])
        
        detailed_file = None
        if detailed_output:
            detailed_file = open(detailed_output, 'w')
            detailed_file.write("sequence_id,predicted_genome,weights,weighted_ranks\n")
        
        for record in SeqIO.parse(reads_file, "fasta"):
            num_reads += 1
            sequence = str(record.seq).upper()
            
            try:
                best_genome_id, method_info = model.classify_sequence(sequence)
                assigned_reads[best_genome_id].append(record.id)
                writer.writerow([record.id, model.genome_names[best_genome_id]])
                
                if detailed_file:
                    weights_str = str(method_info['weights']).replace(',', ';')
                    ranks_str = str(method_info['weighted_ranks']).replace(',', ';')
                    detailed_file.write(f"{record.id},{model.genome_names[best_genome_id]},\"{weights_str}\",\"{ranks_str}\"\n")
                    
            except Exception as e:
                print(f"Error processing {record.id}: {e}")
                # Fallback to first model
                k = model.k_values[0]
                best_genome_id, _ = model.models[k].classify_sequence(sequence)
                assigned_reads[best_genome_id].append(record.id)
                writer.writerow([record.id, model.genome_names[best_genome_id]])
        
        if detailed_file:
            detailed_file.close()
    
    return assigned_reads, num_reads


def evaluate_accuracy(model, test_file, seq_id_map_file):
    """Evaluate model accuracy with detailed metrics"""
    # Load ground truth
    ground_truth = {}
    genome_name_to_id = {name: i for i, name in enumerate(model.genome_names)}
    
    try:
        with open(seq_id_map_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    seq_id = row[0]
                    genome_name = row[1]
                    if genome_name in genome_name_to_id:
                        ground_truth[seq_id] = genome_name_to_id[genome_name]
    except Exception as e:
        print(f"Error reading ground truth: {e}")
        return 0, 0, 0, 0, 0, 0
    
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    for record in SeqIO.parse(test_file, "fasta"):
        if record.id in ground_truth:
            total += 1
            sequence = str(record.seq).upper()
            
            try:
                genome_id, _ = model.classify_sequence(sequence)
                y_true.append(ground_truth[record.id])
                y_pred.append(genome_id)
                
                if genome_id == ground_truth[record.id]:
                    correct += 1
            except Exception as e:
                print(f"Error classifying {record.id}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate precision, recall, and F1-score
    if len(y_true) > 0 and len(y_pred) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    else:
        precision, recall, f1 = 0, 0, 0
    
    return accuracy, correct, total, precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Weighted K-Markov Models')
    parser.add_argument('--genomes', type=str, required=True, help='Path to directory containing genome FASTA files')
    parser.add_argument('--reads', type=str, required=True, help='Reads in FASTA file that need to be assigned')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file with classifications')
    parser.add_argument('--detailed_output', type=str, help='Path to detailed output file')
    parser.add_argument('--k', type=int, nargs='+', default=[3, 5, 7], help='K-mer sizes for ensemble')
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode')
    parser.add_argument('--seq_id_map', type=str, help='Path to sequence ID mapping file (required for evaluation)')
    parser.add_argument('--time', action='store_true', help='Track execution time')
    parser.add_argument('--assignment_summary', type=str, help='Path to output log file')
    
    args = parser.parse_args()

    start_time = time.time()
    
    # Set up logging
    assignment_summary = None
    if args.assignment_summary:
        assignment_summary = open(args.assignment_summary, 'w')
    
    def log_print(message):
        """Print to both console and log file if specified"""
        print(message)
        if assignment_summary:
            assignment_summary.write(message + '\n')
            assignment_summary.flush()
    
    log_print(f"Using adaptive weighting ensemble method")
    log_print(f"K values: {args.k}")
    
    # Train model
    model, num_genomes = train_model(args.genomes, args.k)
    
    # Predict
    assigned_reads, num_reads = predict_sequences(model, args.reads, args.output_csv, args.detailed_output)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print execution summary
    log_print("\n" + "="*50)
    log_print("EXECUTION SUMMARY")
    log_print("="*50)
    log_print(f"Execution time: {elapsed_time:.2f} seconds")
    log_print(f"Number of genome files: {num_genomes}")
    log_print(f"Number of reads: {num_reads}")
    log_print(f"K-mer sizes: {args.k}")
    
    # Get normalized weights
    normalized_weights = model.get_normalized_weights()
    log_print(f"Normalized weights: {normalized_weights}")
    
    # Evaluate if ground truth is provided
    if args.eval and args.seq_id_map:
        accuracy, correct, total, precision, recall, f1 = evaluate_accuracy(model, args.reads, args.seq_id_map)
        log_print(f"Accuracy on test data: {accuracy:.4f} ({correct}/{total})")
        log_print(f"Macro Precision: {precision:.4f}")
        log_print(f"Macro Recall:    {recall:.4f}")
        log_print(f"Macro F1-score:  {f1:.4f}")
    
    # Print assignment statistics
    log_print(f"Total number of reads: {num_reads}")
    log_print(f"Number of assigned reads: {sum(len(reads) for reads in assigned_reads.values())}")
    log_print("Assigned reads by genome:")
    
    # Sort by genome name for consistent output
    for genome_id in sorted(assigned_reads.keys()):
        genome_name = model.genome_names[genome_id]
        read_count = len(assigned_reads[genome_id])
        log_print(f"{genome_name}: {read_count} reads")
    
    log_print("="*50)
    log_print(f"Results saved to {args.output_csv}")
    if args.detailed_output:
        log_print(f"Detailed results saved to {args.detailed_output}")
    
    # Close log file if opened
    if assignment_summary:
        assignment_summary.close()
        print(f"Log saved to {args.assignment_summary}")