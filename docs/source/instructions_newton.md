# Using GPUs on Newton

Steps to get paicos working on Newton, including GPU-code.

## Option 1 (running from terminal)

### Install paicos and turbocluster

Follow the installation instructions [here](https://turbocluster.readthedocs.io/en/latest/installation.html).

### Start an interactive gpu-session
```
srun -p a100 --gres=gpu:1 -n 1 -c 1 --time=1:00:00 --mem=80gb --pty /bin/bash
```

### Activate the conda environment
```
conda activate turbocluster
```

## Option 2 (run a notebook)

First follow the installation instructions as above, then
```
conda activate turbocluster
conda install jupyter 
pip install jupyterlab-widgets
```

Now get a GPU
```
srun -p a100 --gres=gpu:1 -n 1 -c 1 --time=1:00:00 --mem=80gb --pty /bin/bash 
```
Start the jupyter server on the GPU:
```
conda activate turbocluster
jupyter notebook --no-browser 
```

Check which GPU you are on (in this example it is ngpu050), then SSH onto it (port numbers will likely differ in your case)
```
ssh  -L XXXX:localhost:XXXX -J  username@login.aip.de,username@obelisk,username@nnewl3.cls ngpu050
```

Now open the jupyter server in a browser, e.g., go to http://127.0.0.1:XXXX/tree
