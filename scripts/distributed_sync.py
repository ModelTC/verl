"""This script is used for cross-node synchronization to simplify the process 
of establishing a Ray cluster on the SenseCore.

Example:
```Shell
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes $WORLD_SIZE \
    --nproc-per-node 1 --node-rank $RANK --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT scripts/distributed_sync.py
if [ "$RANK" = "0" ]; then
    ray start --head --disable-usage-stats
    sleep 10s # wait for workers' connection
    ray status # check the workers' connection

    {your program}

    ray stop --force
else
    ray start --address $MASTER_ADDR:6379 --block
fi
```
"""
import torch.distributed


torch.distributed.init_process_group(backend="gloo")
torch.distributed.barrier()
torch.distributed.destroy_process_group()
