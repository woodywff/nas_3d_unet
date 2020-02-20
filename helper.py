import subprocess
import numpy as np

# def get_gpus_memory_info():
#     '''
#     Return the GPU with the largest free memory.
#     '''
#     rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
#     rst = rst.strip().split('\n')
#     memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
#     max_id = int(np.argmax(memory_available))
#     return max_id, memory_available