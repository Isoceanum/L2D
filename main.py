# main.py

import os
from train import train_baseline, utils

from evaluate import evaluate_model


def baseline_nominal():
    cwd = os.getcwd()  # e.g., /.../phase_1/code
    root_dir = os.path.dirname(cwd)  # one level up → /.../phase_1
    outputs_dir = os.path.join(root_dir, "outputs")  # → /.../phase_1/outputs
    run_dir = utils.create_output_dir(outputs_dir, method="ppo", variant="nominal")
    
    train_baseline.run_nominal(run_dir)

def baseline_perturbation():
    cwd = os.getcwd()  # e.g., /.../phase_1/code
    root_dir = os.path.dirname(cwd)  # one level up → /.../phase_1
    outputs_dir = os.path.join(root_dir, "outputs")  # → /.../phase_1/outputs
    run_dir = utils.create_output_dir(outputs_dir, method="ppo", variant="perturbation")
    
    train_baseline.run_perturbation(run_dir)

def eval():
    cwd = os.getcwd()
    root_dir = os.path.dirname(cwd)
    outputs_dir = os.path.join(root_dir, "outputs")

    # Point to the specific run you want to evaluate
    method = "ppo"
    variant = "nominal" # "perturbation" "nominal"
    run_id = "000000"

    run_dir = os.path.join(outputs_dir, method, variant, run_id)
    model_path = os.path.join(run_dir, "model.zip")
    output_csv = os.path.join(run_dir, "eval_summary.csv")

    # Evaluate
    evaluate_model.evaluate(model_path, output_csv, n_episodes=100)
    
def nagabandi():
    pass

    
def main():
    #baseline_nominal()
    #baseline_perturbation()
    #eval()
    nagabandi()

    
if __name__ == "__main__":
    main()
