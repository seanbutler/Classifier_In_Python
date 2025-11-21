"""
Study class for organizing and running experiment studies
"""

import csv
import os
from datetime import datetime
from config_manager import load_config, update_config, save_config
from experiment import run_experiment


class Study:
    """
    A Study encapsulates a collection of experiments with common configuration
    and provides methods for running, tracking, and analyzing results.
    """
    
    def __init__(self, name, base_config=None, description=""):
        """
        Initialize a new study
        
        Args:
            name: Name of the study (used for output files)
            base_config: Base configuration dictionary (loads from config.json if None)
            description: Optional description of the study's purpose
        """
        self.name = name
        self.description = description
        self.base_config = base_config if base_config else load_config('config.json')
        self.experiments = []
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create study-specific output directory
        self.output_dir = f'outputs/{self.name}_{self.timestamp}'
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.output_dir}/configs', exist_ok=True)
        os.makedirs(f'{self.output_dir}/results', exist_ok=True)
        
    def add_experiment(self, name, **params):
        """
        Add an experiment to the study
        
        Args:
            name: Name of the experiment
            **params: Parameter overrides (e.g., hidden_layers=[128, 64], learning_rate=0.01)
        """
        exp = {'name': name, **params}
        self.experiments.append(exp)
        
    def add_experiments(self, experiments):
        """
        Add multiple experiments to the study
        
        Args:
            experiments: List of experiment dictionaries with 'name' and parameters
        """
        self.experiments.extend(experiments)
        
    def run(self, verbose=True, save_configs=True, save_results=True):
        """
        Run all experiments in the study
        
        Args:
            verbose: Whether to print detailed progress
            save_configs: Whether to save individual experiment configs
            save_results: Whether to save results to CSV
            
        Returns:
            List of result dictionaries
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Study: {self.name}")
            if self.description:
                print(f"Description: {self.description}")
            print(f"Total Experiments: {len(self.experiments)}")
            print(f"{'='*80}\n")
        
        self.results = []
        
        for i, exp_params in enumerate(self.experiments, 1):
            # Create a copy of base config
            config = {k: v.copy() if isinstance(v, dict) else v 
                     for k, v in self.base_config.items()}
            
            # Get experiment name
            exp_name = exp_params.get('name', f'exp_{i}')
            params_copy = exp_params.copy()
            params_copy.pop('name', None)
            
            # Update config with experiment parameters
            config = update_config(config, **params_copy)
            
            # Pass study output directory to experiment
            config['study_output_dir'] = self.output_dir
            
            # Save this experiment's config if requested
            if save_configs:
                save_config(config, f'{self.output_dir}/configs/config_{exp_name}.json')
            
            # Run the experiment
            if verbose:
                print(f"\n[{i}/{len(self.experiments)}] Running: {exp_name}")
            
            results = run_experiment(config, 
                                    experiment_name=f"{self.name}_{exp_name}", 
                                    verbose=verbose)
            
            # Add experiment parameters to results
            results['parameters'] = params_copy
            results['study_name'] = self.name
            
            self.results.append(results)
        
        # Save results to CSV
        if save_results and self.results:
            csv_filename = f'{self.output_dir}/results/study_{self.name}.csv'
            self._save_to_csv(csv_filename)
            if verbose:
                print(f"\nResults saved to: {csv_filename}")
        
        # Print summary
        if verbose:
            self.print_summary()
        
        return self.results
    
    def _save_to_csv(self, filename):
        """Save study results to CSV file"""
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'experiment_name',
                'hidden_layers',
                'learning_rate',
                'batch_size',
                'momentum',
                'epochs',
                'final_test_accuracy',
                'final_validation_accuracy',
                'best_test_accuracy',
                'best_validation_accuracy',
                'final_loss',
                'min_loss'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'experiment_name': result['experiment_name'],
                    'hidden_layers': str(result['config']['model']['hidden_layers']),
                    'learning_rate': result['config']['training']['learning_rate'],
                    'batch_size': result['config']['training']['batch_size'],
                    'momentum': result['config']['training']['momentum'],
                    'epochs': result['config']['training']['epochs'],
                    'final_test_accuracy': f"{result['final_test_accuracy']:.2f}",
                    'final_validation_accuracy': f"{result['final_validation_accuracy']:.2f}",
                    'best_test_accuracy': f"{result['best_test_accuracy']:.2f}",
                    'best_validation_accuracy': f"{result['best_validation_accuracy']:.2f}",
                    'final_loss': f"{result['final_loss']:.4f}" if result['final_loss'] else 'N/A',
                    'min_loss': f"{result['min_loss']:.4f}" if result['min_loss'] else 'N/A'
                }
                writer.writerow(row)
    
    def print_summary(self, sort_by='best_test_accuracy'):
        """
        Print a summary of study results
        
        Args:
            sort_by: Metric to sort by ('best_test_accuracy', 'final_test_accuracy', etc.)
        """
        
        if not self.results:
            print("No results to display. Run the study first.")
            return
        
        print(f"\n{'='*80}")
        print(f"STUDY SUMMARY: {self.name}")
        print(f"{'='*80}")
        
        # Sort results
        sorted_results = sorted(self.results, 
                               key=lambda x: x.get(sort_by, 0), 
                               reverse=True)
        
        print(f"\n{'Rank':<6} {'Name':<20} {'Architecture':<20} {'LR':<10} {'BS':<6} {'Epochs':<8} {'Best Acc':<10}")
        print(f"{'-'*90}")
        
        for i, result in enumerate(sorted_results, 1):
            name = result['experiment_name'].replace(f"{self.name}_", "")
            arch = str(result['config']['model']['hidden_layers'])
            lr = result['config']['training']['learning_rate']
            bs = result['config']['training']['batch_size']
            epochs = result['config']['training']['epochs']
            acc = result['best_test_accuracy']
            
            print(f"{i:<6} {name:<20} {arch:<20} {lr:<10} {bs:<6} {epochs:<8} {acc:.2f}%")
        
        print(f"\n{'='*90}\n")
    
    def get_best_result(self, metric='best_test_accuracy'):
        """
        Get the best performing experiment
        
        Args:
            metric: Metric to compare ('best_test_accuracy', 'final_test_accuracy', etc.)
            
        Returns:
            Result dictionary of best experiment
        """
        if not self.results:
            return None
        
        return max(self.results, key=lambda x: x.get(metric, 0))
    
    def get_results_dataframe(self):
        """
        Get results as a pandas-compatible dictionary (if pandas is available)
        
        Returns:
            Dictionary suitable for pd.DataFrame(dict)
        """
        if not self.results:
            return {}
        
        data = {
            'experiment_name': [],
            'hidden_layers': [],
            'learning_rate': [],
            'batch_size': [],
            'momentum': [],
            'epochs': [],
            'final_test_accuracy': [],
            'final_validation_accuracy': [],
            'best_test_accuracy': [],
            'best_validation_accuracy': [],
            'final_loss': [],
            'min_loss': []
        }
        
        for result in self.results:
            data['experiment_name'].append(result['experiment_name'])
            data['hidden_layers'].append(str(result['config']['model']['hidden_layers']))
            data['learning_rate'].append(result['config']['training']['learning_rate'])
            data['batch_size'].append(result['config']['training']['batch_size'])
            data['momentum'].append(result['config']['training']['momentum'])
            data['epochs'].append(result['config']['training']['epochs'])
            data['final_test_accuracy'].append(result['final_test_accuracy'])
            data['final_validation_accuracy'].append(result['final_validation_accuracy'])
            data['best_test_accuracy'].append(result['best_test_accuracy'])
            data['best_validation_accuracy'].append(result['best_validation_accuracy'])
            data['final_loss'].append(result['final_loss'])
            data['min_loss'].append(result['min_loss'])
        
        return data


# Example usage
if __name__ == "__main__":
    
    # Create a study
    study = Study(
        name="architecture_comparison",
        description="Comparing different network architectures on MNIST"
    )
    
    # Add experiments
    # study.add_experiment('baseline', hidden_layers=[64, 32], learning_rate=0.001, batch_size=64, epochs=10)
    # study.add_experiment('wider', hidden_layers=[128, 64], learning_rate=0.001, batch_size=64, epochs=10)
    # study.add_experiment('deeper', hidden_layers=[64, 32, 16], learning_rate=0.001, batch_size=64, epochs=10)
    # study.add_experiment('single', hidden_layers=[128], learning_rate=0.001, batch_size=64, epochs=10)

    # study.add_experiment('batch16', hidden_layers=[64, 32], learning_rate=0.001, batch_size=16, epochs=20)
    # study.add_experiment('batch32', hidden_layers=[64, 32], learning_rate=0.001, batch_size=32, epochs=20)
    # study.add_experiment('batch64', hidden_layers=[64, 32], learning_rate=0.001, batch_size=64, epochs=20)
    # study.add_experiment('batch128', hidden_layers=[64, 32], learning_rate=0.001, batch_size=128, epochs=20)

    study.add_experiment('layers_1', hidden_layers=[32], learning_rate=0.001, batch_size=64, epochs=35)
    study.add_experiment('layers_2', hidden_layers=[32, 32], learning_rate=0.001, batch_size=64, epochs=35)
    study.add_experiment('layers_3', hidden_layers=[32, 32, 32], learning_rate=0.001, batch_size=64, epochs=35)
    study.add_experiment('layers_4', hidden_layers=[32, 32, 32, 32], learning_rate=0.001, batch_size=64, epochs=35)

    # Or add multiple at once
    # additional_experiments = [
    #     {'name': 'tiny', 'hidden_layers': [32], 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 10},
    #     {'name': 'huge', 'hidden_layers': [512, 256, 128], 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 10},
    # ]
    # study.add_experiments(additional_experiments)
    
    # Run the study
    results = study.run(verbose=True)
    
    # Get best result
    best = study.get_best_result()
    print(f"\nBest experiment: {best['experiment_name']} with {best['best_test_accuracy']:.2f}% accuracy")
