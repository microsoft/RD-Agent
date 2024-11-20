#!/bin/sh
cat << "EOF" > /dev/null
Given a directory with *.env files.  Run each one.

usage for example:

  1) directly run command without extra shared envs
  ./run_envs.sh -d <dir_to_*.envfiles> -j <number of parallel process> -- <command>

  2) load shared envs `.env` before running command with different envs.
  dotenv run -- ./run_envs.sh -d <dir_to_*.envfiles> -j <number of parallel process> -- <command>

EOF

# Function to display usage
usage() {
  echo "Usage: $0 -d <dir_to_*.envfiles> -j <number of parallel process> -- <command>"
  exit 1
}

# Parse command line arguments
while getopts "d:j:" opt; do
  case $opt in
    d) DIR=$OPTARG ;;
    j) JOBS=$OPTARG ;;
    *) usage ;;
  esac
done

# Shift to get the command
shift $((OPTIND -1))

# Check if directory and jobs are set
if [ -z "$DIR" ] || [ -z "$JOBS" ] || [ $# -eq 0 ]; then
  usage
fi

COMMAND="$@"

# Before running commands
echo "Running experiments with following env files:"
find "$DIR" -name "*.env" -exec echo "{}" \;

# Export and run each .env file in parallel
find "$DIR" -name "*.env" | xargs -n 1 -P "$JOBS" -I {} sh -c "
  set -a
  . {}
  set +a
  $COMMAND
"

