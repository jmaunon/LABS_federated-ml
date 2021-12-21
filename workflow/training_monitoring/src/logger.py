import time
from numbers import Number
from typing import Dict, List

import requests

class Logger:
    def __init__(self, job_id, node_id):
        self.job_id = job_id
        self.node_id = node_id
        self.uri = 'http://logger:9091/metrics/job/%s/node/%s' % (job_id, node_id)

    def log(self, metric: str, value: Number, extra_labels: Dict = None):
        try:
            # parse extra labels and it format to <label_title>=<label_value>
            labels = ''
            if extra_labels:
                label_list = [ '%s="%s"' % (k, v) for k, v in extra_labels.items() ]
                labels = '{%s}' % (','.join(label_list))

            # format data to format <metric_title>{<labels>} <metric_value>
            data = '%s%s %d\n' % (metric, labels, value)

            response = requests.post(self.uri, data=data)
            response.raise_for_status()
        except Exception as err:
            print(err)
            with err.response as res:
                print("[Logger status %d]: %s (%s -> %d %s %s)" % (res.status_code, 
                    res.text, metric, value, labels, self.uri))
    
    def logExecutionTime(self, method):
        def inner(*args, **kwargs):
            start_time = time.time()
            res = method(*args, **kwargs)
            end_time = time.time()
            
            execution_time = (end_time - start_time)
            print(execution_time)
            self.log("method_duration_seconds", execution_time, { "method_name": method.__name__ })
            
            return res

        return inner