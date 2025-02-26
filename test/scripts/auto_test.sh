#!/bin/bash
set -x # show command
set -e # Error on exception

DIR="$(
  cd "$(dirname "$(readlink -f "$0")")" || exit
  pwd -P
)"

deploy() {
  FROM_ENV_PATH=~/repos/RD-Agent/.env

  latest_commit=$(git ls-remote git@github.com:microsoft/RD-Agent.git HEAD | awk '{print substr($1, 1, 10)}')
  echo "Latest commit hash: $latest_commit"

  REPO_FOLDER=$DIR/RD-Agent-$latest_commit

  cd $DIR

  git clone git@github.com:microsoft/RD-Agent.git $REPO_FOLDER
  cd $REPO_FOLDER
  cp $FROM_ENV_PATH ./.env
  conda create -n latest_commit python=3.10 -y
  . $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit
  conda activate latest_commit
  make dev

  cd $REPO_FOLDER
  wget https://github.com/SunsetWolf/rdagent_resource/releases/download/kaggle_data/kaggle_data.zip
  unzip kaggle_data.zip -d ./kaggle_data
}

batch_test() {
  cd $DIR
  mkdir -p stdout/
cat << "EOF" > $(ls -td $DIR/RD-Agent-* | head -n 1)/cmds
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent fin_factor" ../stdout/fin_factor
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent fin_model" ../stdout/fin_model
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent med_model" ../stdout/med_model
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent fin_factor_report --report_folder=git_ignore_folder/reports" ../stdout/fin_factor_report
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent general_model 'https://arxiv.org/pdf/2210.09789'" ../stdout/general_model
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition covid19-global-forecasting-week-1" ../stdout/covid19-global-forecasting-week-1
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition digit-recognizer" ../stdout/digit-recognizer
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition feedback-prize-english-language-learning" ../stdout/feedback-prize-english-language-learning
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition forest-cover-type-prediction" ../stdout/forest-cover-type-prediction
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition optiver-realized-volatility-prediction" ../stdout/optiver-realized-volatility-prediction
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s3e11" ../stdout/playground-series-s3e11
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s3e14" ../stdout/playground-series-s3e14
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s3e16" ../stdout/playground-series-s3e16
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s3e26" ../stdout/playground-series-s3e26
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s4e5" ../stdout/playground-series-s4e5
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s4e8" ../stdout/playground-series-s4e8
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition playground-series-s4e9" ../stdout/playground-series-s4e9
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition sf-crime" ../stdout/sf-crime
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition spaceship-titanic" ../stdout/spaceship-titanic
script -c ". $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate latest_commit && which python && rdagent kaggle --competition statoil-iceberg-classifier-challenge" ../stdout/statoil-iceberg-classifier-challenge
EOF
  # start the commands parallelly with  `cat cmds | tmuxr -p 20 -s rdagent-test`
}

clean() {
  cd $DIR
  rm -rf RD-Agent-*
  rm -rf stdout
  conda remove --name latest_commit --all -y
}

$1
