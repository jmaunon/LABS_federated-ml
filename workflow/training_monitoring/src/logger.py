import requests

class Logger:
    def __init__(self, job_id, node_id):
        self.job_id = job_id
        self.node_id = node_id
        self.uri = 'http://logger:9091/metrics/job/%s/node/%s' % (job_id, node_id)

    def log(self, param, value):
        try:
            data = '%s %s' % (param, value)
            requests.post(self.uri, data=data)
        except Exception as err:
            print(err)