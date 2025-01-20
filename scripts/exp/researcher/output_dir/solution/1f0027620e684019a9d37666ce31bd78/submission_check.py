import pandas as pd
from pathlib import Path

# Check if the sample submission file exists
if not Path("/kaggle/input/sample_submission.csv").exists():
    exit(0)

sample_submission = pd.read_csv('/kaggle/input/sample_submission.csv')
our_submission = pd.read_csv('submission.csv')

success = True
for col in sample_submission.columns:
    if col not in our_submission.columns:
        success = False
        print(f'Column {col} not found in submission.csv')

if success:
    print('submission.csv is valid.')