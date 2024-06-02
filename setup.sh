

# install k2
pip install k2==1.24.4.dev20240223+cpu.torch1.13.1 -f https://k2-fsa.github.io/k2/cpu.html

# install icefall
cd /tmp
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=/tmp/icefall:$PYTHONPATH

