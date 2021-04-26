import numpy as np

import datetime
import time

class NumericalParameter:
    def __init__(self, name: str, *fields):
        fields = fields[0]
        self.name = name
        self.field_names = []
        self.fields = {}

        for f in fields:
            if isinstance(f, tuple):
                fn, ft = f
            else:
                fn, ft = f, float

            self.field_names.append(fn)
            self.fields[fn] = np.zeros((0,), dtype=ft)

    def append_value(self, *values):
        assert len(values) == len(self.field_names), "nb. values and nb. fields do not match!"
        for i, v in enumerate(values):
            fn = self.field_names[i]
            self.fields[fn] = np.append(self.fields[fn], v)

class Logger:
    def __init__(self):
        self.parameters = {}

    def create_parameter(self, name, *field_names):
        self.parameters[name] = NumericalParameter(name, field_names)

    def get_parameter(self, name):
        return self.parameters[name]
    get = get_parameter

if __name__ == '__main__':
    print("Aloha, World!")

    logger = Logger()
    logger.create_parameter('loss', ('epoch', int), 'mse_loss')

    for i in range(100):
        logger.get('loss').append_value(i, 100.0 - i)

    print(logger.get('loss').fields)
