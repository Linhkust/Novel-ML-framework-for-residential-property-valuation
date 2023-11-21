"""client.py should be run on Client Node"""
import time, queue
import pandas as pd
from multiprocessing.managers import BaseManager
from regressor import Regressor

'''
   Task queue configuration
   '''
data = pd.read_csv('paper.csv')
poi_threshold = [300, 500]
gsv_threshold = [500, 1000]
rs_threshold = [500, 1000]
cnn_type = ['GoogleNet', 'AlexNet', 'VGG16', 'ResNet101']

reg = Regressor(data, poi_threshold, gsv_threshold, rs_threshold, cnn_type)

# Get the queue
BaseManager.register('get_task_queue')
BaseManager.register('get_result_queue')

# Identify the Server IP address
server_address = 'Your Server IP Address'

print(f'Connecting to server {server_address}...')

m = BaseManager(address=(server_address, 5000), authkey=b'password')
m.connect()

task = m.get_task_queue()
result = m.get_result_queue()

for i in range(512):
    try:
        pipeline = task.get(timeout=1)

        # computing part
        r = reg.pipelines_fit_single(pipeline)

        time.sleep(1)

        result.put(r)

    except queue.Empty:
        print('task queue is empty.')

print('worker exit.')
