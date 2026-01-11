## Troubleshooting

### Error: "failed to run command 'qrun': No such file or directory"

**Cause**: Qlib dependency is not properly installed.

**Solution**:
1. Install Qlib from source:
pip install numpy
pip install --upgrade cython
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install .

text

2. Verify installation:
which qrun

text

3. Download Qlib data:
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn