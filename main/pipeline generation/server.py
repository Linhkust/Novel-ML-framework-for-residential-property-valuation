"""server.py should be run on Server Node"""

import queue
from multiprocessing.managers import BaseManager
import warnings
import pandas as pd
from regressor import Regressor
import time

warnings.filterwarnings('ignore')

_task = queue.Queue()
_result = queue.Queue()

def task_queue():
    return _task

def result_queue():
    return _result


# Register on the Network so other nodes can visit
BaseManager.register('get_task_queue', callable=task_queue)
BaseManager.register('get_result_queue', callable=result_queue)

# IP address, port and authentication key
manager = BaseManager(address=('Your Server IP Address', 5000), authkey=b'password')

if __name__ == '__main__':
    '''
    Task queue configuration
    '''
    data = pd.read_csv('paper.csv')
    poi_threshold = [300, 500]
    gsv_threshold = [500, 1000]
    rs_threshold = [500, 1000]
    cnn_type = ['GoogleNet', 'AlexNet', 'VGG16', 'ResNet101']

    reg = Regressor(data, poi_threshold, gsv_threshold, rs_threshold, cnn_type)

    # All pipelines
    pipelines = reg.generate_pipelines()

    # Start the computing
    start = time.time()

    manager.start()

    task = manager.get_task_queue()

    result = manager.get_result_queue()

    for pipeline in pipelines:
        task.put(pipeline)

    print('Waiting for computing results...')

    # Pipeline performance training results
    results = []

    # For all pipeline computing
    for i in range(len(pipelines)):
        r = result.get()
        r.insert(0, 'Pipeline_{}'.format(i+1))
        results.append(r)

    manager.shutdown()

    finish = time.time()

    print("%.2f" % (finish-start))
    results = pd.DataFrame(results, columns=['pipeline_id', 'model_type', 'features', 'hyperparameters', 'Time',
                                             'before_r2', 'before_mae', 'before_rmse',
                                             'after_r2', 'after_mae', 'after_rmse',
                                             'improvement'])

    results.to_csv('Pipeline.csv', index=False)


