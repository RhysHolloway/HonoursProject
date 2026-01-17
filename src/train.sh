cd "$(dirname "$0")"
cd "../env/deep-early-warnings-pnas/training_data/"
source ../../bury-venv/bin/activate
source ../../auto/cmds/auto.env.sh
source ./run_single_batch.sh $1 $2