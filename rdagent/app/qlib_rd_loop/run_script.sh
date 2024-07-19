#!/bin/bash

max_retries=1000
count=0
log_file="/home/finco/v-yuanteli/RD-Agent/rdagent/app/qlib_rd_loop/run_script.log"

while [ $count -lt $max_retries ]; do
  echo "$(date) - Attempt $count of $max_retries" >> $log_file
  /home/finco/anaconda3/envs/rdagent/bin/python /home/finco/v-yuanteli/RD-Agent/rdagent/app/qlib_rd_loop/factor_from_report_sh.py >> $log_file 2>&1
  if [ $? -eq 0 ]; then
    echo "$(date) - Script completed successfully on attempt $count" >> $log_file
    break
  fi
  count=$((count + 1))
  echo "$(date) - Restarting script after crash... Attempt $count of $max_retries" >> $log_file
done

if [ $count -ge $max_retries ]; then
  echo "$(date) - Script failed after $max_retries attempts." >> $log_file
else
  echo "$(date) - Script completed successfully." >> $log_file
fi

# chmod +x /home/finco/v-yuanteli/RD-Agent/rdagent/app/qlib_rd_loop/run_script.sh