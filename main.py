# main.py

import os
from nagabandi.evaluate_mpc import evaluate_mpc
from nagabandi.train_dynamics import train_dynamics_model
from nagabandi.data_collector import collect_transitions
from train import train_baseline, utils

from evaluate import evaluate_model

def output_path(dir: str):
    cwd = os.getcwd()
    root_dir = os.path.dirname(cwd)
    return os.path.join(root_dir, dir)
    

def baseline_nominal():
    outputs_dir = output_path("outputs")
    run_dir = utils.create_output_dir(outputs_dir, method="ppo", variant="nominal")
    train_baseline.run_nominal(run_dir)

def baseline_perturbation():
    outputs_dir = output_path("outputs")
    run_dir = utils.create_output_dir(outputs_dir, method="ppo", variant="perturbation")
    train_baseline.run_perturbation(run_dir)

def eval():
    outputs_dir = output_path("outputs")
    # Point to the specific run you want to evaluate
    method = "ppo"
    variant = "nominal" # "perturbation" "nominal"
    run_id = "000000"

    run_dir = os.path.join(outputs_dir, method, variant, run_id)
    model_path = os.path.join(run_dir, "model.zip")
    output_csv = os.path.join(run_dir, "eval_summary.csv")

    # Evaluate
    evaluate_model.evaluate(model_path, output_csv, n_episodes=100)
    
def nagabandi_step_1(outputs_dir):
    num_episodes = 10
    collect_transitions(num_episodes=num_episodes, save_path=outputs_dir)
    
def nagabandi_step_2(outputs_dir):
    train_dynamics_model(outputs_dir)
    
def nagabandi_step_3(outputs_dir):
    model_path = os.path.join(outputs_dir, "dynamics_model.pth")
    evaluate_mpc(model_path=model_path, n_episodes=5)
    
def nagabandi():
    outputs_dir = output_path("nagabandi")
    nagabandi_step_1(outputs_dir)
    nagabandi_step_2(outputs_dir)
    nagabandi_step_3(outputs_dir)

def main():    
    #baseline_nominal()
    #baseline_perturbation()
    #eval()
    nagabandi()

if __name__ == "__main__":
    main()
