

# create baseline
```bash

# C_LIST=(measured-jackal wondrous-bluegill optimum-sole noted-whale great-guppy amusing-glider)
C_LIST=("deciding-cod" "exotic-frog" "civil-reindeer" "selected-worm" "prepared-salmon")


AMLT_DIR=/Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/
PROC_AMLT_DIR=/Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt_processed/
h12d=$PROC_AMLT_DIR/12h/
mkdir -p $h12d

ls $PROC_AMLT_DIR

for c in ${C_LIST[@]} ; do
  # ls $AMLT_DIR/${c}/
  cp -r $AMLT_DIR/${c}/ $h12d/${c}
done

# for c in ${C_LIST[@]} ; do
# cp -r /Data/home/xiaoyang/repos/JobAndExp/amlt_project/amlt/
for c in ${C_LIST[@]} ; do
  mydotenv.sh python rdagent/log/mle_summary.py summary --log_folder=$h12d/$c/combined_logs --hours=12
done


for c in ${C_LIST[@]} ; do
  mydotenv.sh python rdagent/log/mle_summary.py summary --log_folder=$h12d/$c/combined_logs --hours=13
done

```
