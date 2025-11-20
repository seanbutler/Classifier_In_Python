"""
Main application entry point - simple wrapper for running experiments
"""

from config_manager import load_config
from experiment import run_experiment

############################################################

def main():
    """Main function - loads config and runs a single experiment"""
    
    # Load configuration
    config = load_config('config.json')
    
    # Run the experiment
    results = run_experiment(config, experiment_name='default', verbose=True)
    
    return results

############################################################

if __name__ == "__main__":
    main()
