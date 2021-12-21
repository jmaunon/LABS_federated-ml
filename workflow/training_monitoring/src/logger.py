import time
from numbers import Number
from typing import Dict, Callable

import requests

class Logger:
    """
    A simple class to send metric to Prometheus via PushGateway service. 
    It uses 'requests' library to perform the HTTP calls.

    Attributes
    ----------
    job_id : str
        the identifier of the Prometheus job
    node_id : str
        the identifier of the Node label in Prometheus. It identifies the 
        silo where the trace occurs (0: server; 1: client_a; 2: client_b).

    Methods
    -------
    log(metric: str, value: Number, extra_labels: Dict = None)
        send a trace for the metric defined with the provided value, also 
        append extra labels if they are provided.

    Decorators
    ----------
    logExecutionTime:
        send a trace with the label 'method_duration_seconds' that includes
        the excution elapsed time of the decorated method with its name as
        extra label.
    """

    job_id: str
    node_id: str

    def __init__(self, job_id, node_id):
        # Store prometheus job and node ID
        self.job_id = job_id
        self.node_id = node_id

        # Format the pushgateway URI 
        self.uri = 'http://logger:9091/metrics/job/%s/node/%s' % (job_id, node_id)

    def log(self, metric: str, value: Number, extra_labels: Dict = None) -> None:
        try:
            # Parse extra labels and it format to <label_title>=<label_value>
            labels = ''
            if extra_labels:
                label_list = [ '%s="%s"' % (k, v) for k, v in extra_labels.items() ]
                labels = '{%s}' % (','.join(label_list))

            # Format data to format <metric_title>{<labels>} <metric_value>
            data = '%s%s %d\n' % (metric, labels, value)

            response = requests.post(self.uri, data=data)
            response.raise_for_status()
            return
        except Exception as err:
            # Print generic exception
            print(err)

            # Print complex exception when it is a HTTP exception
            with err.response as res:
                print("[Logger status %d]: %s (%s -> %d %s %s)" % (res.status_code, 
                    res.text, metric, value, labels, self.uri))
    
    def logExecutionTime(self, method: Callable) -> Callable:
        name = method.__name__
        # Create a decorator to measure the execution time of any function
        def inner(*args, **kwargs):
            # Get the initial time, execute the decorated method and get the 
            # result and the final time.
            start_time = time.time()
            res = method(*args, **kwargs)
            end_time = time.time()
            
            # Calc elapsed time in seconds, register the trace and return 
            # the decorated result.
            execution_time = (end_time - start_time)
            # WORKAROUND: Prometheus does not format seconds with many decimals 
            # correctly.Convert time unit to microseconds and append an extra 
            # label to the trace that includes the time unit.
            execution_time *= 1000000
            extra_labels = { "time_unit": "us" }

            # WORKAROUND: Prometheus groups metrics with the same label by time.
            # Including the method name into the label prevents this aggregation.
            label = "%s_method_duration" % method.__name__
            self.log(label, execution_time, extra_labels)

            # Return the original result of the method execution.
            return res

        # Return the inner decorator function
        return inner