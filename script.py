import argparse
import os
import subprocess

# Parse command-line options
parser = argparse.ArgumentParser(description="Build or run a Docker container")
parser.add_argument(
    "--build", action="store_true", help="Build the Docker image"
)
parser.add_argument(
    "--run", action="store_true", help="Run the Docker container"
)
args = parser.parse_args()

IMAGE_NAME = "jet-physics"
WORK_HOME = "/usr/local/jet-physics-and-machine-learning"

# Build the Docker image if the --build option is set
if args.build:
    subprocess.run(["docker", "build", "-t", f"{IMAGE_NAME}", "."])

# Run the Docker container if the --run option is set
if args.run:
    current_dir = os.getcwd()
    subprocess.run(
        [
            "docker",
            "run",
            "-it",
            "-v",
            f"{current_dir}:/{WORK_HOME}",
            f"{IMAGE_NAME}",
            "bash",
        ]
    )
