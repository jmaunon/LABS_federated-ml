import requests

class Logger:
    def __init__(self, job_id, node_id):
        self.job_id = job_id
        self.node_id = node_id
        self.uri = 'http:localhost:9091/metrics/job/%d/node/%d' % (job_id, node_id)

    def log(self, node_id, param, value):
        try:
            data = '%s %s' % (param, value)
            requests.posts(self.uri, data=data)
        except Exception as err:
            print(err)