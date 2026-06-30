import argparse
from pathlib import Path

from bevad_sim.simulation.carla.eval import evaluate_single_scenario

AGENT_MODULE = "bevad.agent.bevad_agent"
DEFAULT_TIMEOUT = 120  # seconds
SENSOR_CONFIG = "bevad/b2d_agent.json"


def closed_loop_simulation(
    config_path: str,
    checkpoint_path: str,
    route_path: Path,
    output_dir: Path,
) -> Path:
    agent_config = {"cfg": f"{config_path}+{checkpoint_path}"}
    episode_dir = evaluate_single_scenario(
        route_file=route_path,
        sensor_config=SENSOR_CONFIG,
        agent_module_name=AGENT_MODULE,
        agent_config=agent_config,
        permit_infractions=True,
        timeout=DEFAULT_TIMEOUT,
        output_dir=output_dir,
        downsample_by_n=1,
        fixed_seed=2000,
    )
    return episode_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed-loop CARLA simulation with a BevAD agent."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bevad/bevad/configs/cvpr/scaling_diffusion.py",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bevad-m.ckpt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--route",
        type=Path,
        default=Path("data/b2d-xml/b2d-24224.xml"),
        help="Path to the XML route file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory to write episode recordings to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    episode_dir = closed_loop_simulation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        route_path=args.route,
        output_dir=args.output_dir,
    )
    print("Data written to:", episode_dir)
