import argparse
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

# Build the Docker image if the --build option is set
if args.build:
    subprocess.run(["docker", "build", "-t", f"{IMAGE_NAME}", "."])

# Run the Docker container if the --run option is set
if args.run:
    subprocess.run(["docker", "run", "-it", f"{IMAGE_NAME}", "bash"])
