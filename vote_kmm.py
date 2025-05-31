import os
import numpy as np
from Bio import SeqIO
from collections import defaultdict, Counter
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


class MajorityVotingKMM:
    def __init__(self, k_values=[3, 5, 7]):
        self.k_values = sorted(k_values)  # Sort to ensure consistent ordering
        self.models = {}
        self.genome_names = []
        self.voting_stats = {
            'unanimous': 0,
            'majority': 0,
            'tie_breaker': 0,
            'total': 0
        }
        
        # Initialize models for each k value
        for k in self.k_values:
            self.models[k] = KMarkovModel(k)
    
    def train(self, genome_files):
        """Train all base models"""
        print(f"Training {len(self.k_values)} base models...")
        
        for k in self.k_values:
            print(f"Training k={k} model...")
            self.models[k].train(genome_files)
        
        self.genome_names = self.models[self.k_values[0]].genome_names
        print("Base models training completed.")
    
    def classify_sequence(self, sequence):
        """
        Classify sequence using majority voting:
        1. If 2+ models agree, choose that genome
        2. If all models disagree, choose prediction from largest k model
        """
        # Get predictions from all models
        predictions = {}
        all_scores = {}
        
        for k in self.k_values:
            best_genome_id, best_score = self.models[k].classify_sequence(sequence)
            predictions[k] = best_genome_id
            all_scores[k] = self.models[k].get_all_scores(sequence)
        
        # Count votes for each genome
        vote_counts = Counter(predictions.values())
        self.voting_stats['total'] += 1
        
        # Determine final prediction based on majority voting
        max_votes = max(vote_counts.values())
        most_voted_genomes = [genome for genome, votes in vote_counts.items() if votes == max_votes]
        
        if max_votes >= 2:
            # Majority vote (2 or 3 models agree)
            final_prediction = most_voted_genomes[0]  # Take the first one if tie in majority
            if max_votes == len(self.k_values):
                decision_type = 'unanimous'
                self.voting_stats['unanimous'] += 1
            else:
                decision_type = 'majority'
                self.voting_stats['majority'] += 1
        else:
            # All models disagree - use largest k model
            largest_k = max(self.k_values)
            final_prediction = predictions[largest_k]
            decision_type = 'tie_breaker'
            self.voting_stats['tie_breaker'] += 1
        
        # Prepare detailed information
        method_info = {
            'predictions': predictions,
            'vote_counts': dict(vote_counts),
            'decision_type': decision_type,
            'final_prediction': final_prediction,
            'all_scores': all_scores
        }
        
        return final_prediction, method_info
    
    def get_voting_statistics(self):
        """Get statistics about voting patterns"""
        total = self.voting_stats['total']
        if total == 0:
            return {}
        
        return {
            'unanimous_percentage': (self.voting_stats['unanimous'] / total) * 100,
            'majority_percentage': (self.voting_stats['majority'] / total) * 100,
            'tie_breaker_percentage': (self.voting_stats['tie_breaker'] / total) * 100,
            'total_sequences': total,
            'raw_counts': dict(self.voting_stats)
        }


def train_model(genomes_dir, k_values=[3, 5, 7]):
    genome_files = [os.path.join(genomes_dir, f) for f in os.listdir(genomes_dir) 
                    if f.endswith('.fa') or f.endswith('.fasta') or f.endswith('.fna')]
    
    model = MajorityVotingKMM(k_values)
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
            detailed_file.write("sequence_id,predicted_genome,decision_type,k3_prediction,k5_prediction,k7_prediction,vote_counts\n")
        
        for record in SeqIO.parse(reads_file, "fasta"):
            num_reads += 1
            sequence = str(record.seq).upper()
            
            try:
                best_genome_id, method_info = model.classify_sequence(sequence)
                assigned_reads[best_genome_id].append(record.id)
                writer.writerow([record.id, model.genome_names[best_genome_id]])
                
                if detailed_file:
                    predictions = method_info['predictions']
                    k_predictions = [str(predictions.get(k, 'N/A')) for k in sorted(model.k_values)]
                    vote_counts_str = str(method_info['vote_counts']).replace(',', ';')
                    
                    detailed_file.write(
                        f"{record.id},"
                        f"{model.genome_names[best_genome_id]},"
                        f"{method_info['decision_type']},"
                        f"{','.join(k_predictions)},"
                        f"\"{vote_counts_str}\"\n"
                    )
                    
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
    parser = argparse.ArgumentParser(description='Majority Voting K-Markov Models')
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
    
    log_print(f"Using majority voting ensemble method")
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
    
    # Get voting statistics
    voting_stats = model.get_voting_statistics()
    if voting_stats:
        log_print(f"Voting Statistics:")
        log_print(f"  Unanimous decisions: {voting_stats['unanimous_percentage']:.1f}% ({voting_stats['raw_counts']['unanimous']} sequences)")
        log_print(f"  Majority decisions:  {voting_stats['majority_percentage']:.1f}% ({voting_stats['raw_counts']['majority']} sequences)")
        log_print(f"  Tie-breaker decisions: {voting_stats['tie_breaker_percentage']:.1f}% ({voting_stats['raw_counts']['tie_breaker']} sequences)")
    
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