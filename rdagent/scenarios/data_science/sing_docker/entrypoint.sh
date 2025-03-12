#!/bin/sh
set -x

DIR="$( cd "$(dirname "$(readlink -f "$0")")" || exit ; pwd -P )"

sudo mkdir -p /mle/ /kaggle/

CURRENT_USER=$(id -un)
sudo chown -R $CURRENT_USER:$CURRENT_USER /workspace/ /mle/ /kaggle/

ls -lat /

cd $DIR/../RD-Agent
mkdir -p log/
git fetch
git checkout ${RD_COMMIT:-ee8d97c52062607cac778b8aeb10769b075a8d11}
make dev
pip install 'litellm[proxy]'
pip install git+https://github.com/you-n-g/litellm@add_mi_cred_pr


cd $DIR/../litellm-srv/
export AZURE_CLIENT_ID
export AZURE_SCOPE=api://trapi/.default
export AZURE_CREDENTIAL=ManagedIdentityCredential
sed -i '/proxy_handler_instance/d' litellm.trapi.yaml  # remove useless handler in production
nohup litellm --config litellm.trapi.yaml &

sleep 10  # wait for litellm to start


cd $DIR/../RD-Agent
script -c "timeout ${RD_TIMEOUT:-24h} python rdagent/app/data_science/loop.py --competition $DS_COMPETITION" log/stdout.${DS_COMPETITION}.log

unset LOG_TRACE_PATH  # avoid make the original log dirty.
python rdagent/log/mle_summary.py grade_summary --log_folder=./log/

tar cf log.tar log

# NOTE: when we have $AMLT_OUTPUT_DIR, maybe we don't have to copy file actively to azure blob now.
# RD_OUTPUT_DIR=${RD_OUTPUT_DIR:-/data/rdagent}/
# mkdir -p $RD_OUTPUT_DIR
# cp -r log.tar $RD_OUTPUT_DIR/${RD_RES_NAME:-log.tar}

cp -r log.tar $AMLT_OUTPUT_DIR/${RD_RES_NAME:-log.tar}

set > $AMLT_OUTPUT_DIR/env
