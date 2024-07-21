# Python Template

A [simple] [general-purpose] Python template 🐍🚀🎉🦕

## HPC Resource Allocation

1. Start new tmux session
    ```bash
    tmux new -s <session-name>
    ```

2. Request interactive session
   For example, to request 24 cores, 2 GPUs, and 32 GB memory:
    ```bash
    srun -A vxc204_aisc -p aisc -c 24 --gres=gpu:2 --mem=32G --pty bash
    ```
## Run on HPC

1. Clone the repository
    ```bash
    git clone -b hpc git@github.com:mohsenhariri/template-python.git
    chmod -R u=rwx,go= template-python
    cd template-python
    ```
    
2. Run the initialization script 
    ```bash
    chmod +x hpc.init
    source hpc.init
    ```

3. Modify or add commands to `hpc.make` file
   
4. Run the `hpc.make` file
    ```bash
    make newCommand
    ```