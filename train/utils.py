import os

def create_output_dir(outputs_dir: str, method: str = "ppo", variant: str = "nominal") -> str:
    """
    Creates a run directory like: {outputs_dir}/{method}/{variant}/000000/

    Args:
        outputs_dir (str): Absolute path to the root outputs folder
        method (str): Algorithm name (e.g., 'ppo', 'grbal')
        variant (str): Experiment variant (e.g., 'nominal', 'perturbation')

    Returns:
        str: Absolute path to the created run directory
    """
    variant_root = os.path.join(outputs_dir, method, variant)
    os.makedirs(variant_root, exist_ok=True)

    # Auto-increment the next run ID
    existing = [d for d in os.listdir(variant_root) if d.isdigit()]
    existing = sorted(int(name) for name in existing)
    next_id = (max(existing) + 1) if existing else 0

    run_name = f"{next_id:06d}"
    run_path = os.path.join(variant_root, run_name)
    os.makedirs(run_path, exist_ok=True)

    return run_path
