from typing import Iterable

from utils.data_manager1 import IncrementalDataManager


class DataManager:
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args or {}
        self._idm = IncrementalDataManager(
            dataset_name=dataset_name,
            initial_classes=init_cls,
            increment_classes=increment,
            shuffle=shuffle,
            seed=seed)
        self.dataset_name = dataset_name

    @property
    def nb_tasks(self) -> int:
        return self._idm.nb_tasks

    def get_task_size(self, task: int) -> int:
        return self._idm.get_task_size(task)

    def get_dataset(self, class_indices: Iterable[int], source: str, mode: str):
        return self._idm.get_dataset(class_indices, source, mode)
