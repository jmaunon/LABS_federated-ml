import time
from numbers import Number
from typing import Dict, List

import requests

class Logger:
    def __init__(self, job_id, node_id):
        # store prometheus job and node ID
        self.job_id = job_id
        self.node_id = node_id

        # format the pushgateway URI 
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
            # print generic exception
            print(err)

            # print complex exception when it is a HTTP exception
            with err.response as res:
                print("[Logger status %d]: %s (%s -> %d %s %s)" % (res.status_code, 
                    res.text, metric, value, labels, self.uri))
    
    def logExecutionTime(self, method):
        # create a decorator to measure the execution time of any function
        def inner(*args, **kwargs):
            # get the initial time, execute the decorated method and get the 
            # result and the final time.
            start_time = time.time()
            res = method(*args, **kwargs)
            end_time = time.time()
            
            # calc elapsed time in seconds, register the trace and return 
            # the decorated result.
            execution_time = (end_time - start_time)
            print(execution_time)
            self.log("method_duration_seconds", execution_time, { "method_name": method.__name__ })
            return res

        # return inner decoration function
        return inner