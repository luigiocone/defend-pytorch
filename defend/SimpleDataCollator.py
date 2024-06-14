from dataclasses import dataclass


@dataclass
class SimpleDataCollator:
    def __call__(self, features):
        return features