import warnings
from tensorboardX import SummaryWriter


class SummaryMetric:
    def __init__(self, name, value, compare_method='max') -> None:
        self._name = name
        self._metric = value
        self.compare_method = compare_method

    def update(self, value):
        if self.compare_method == 'max':
            if value > self._metric:
                self._metric = value
        elif self.compare_method == 'min':
            if value < self._metric:
                self._metric = value
        else:
            raise NotImplementedError
        return self._metric

    @property
    def best_metric(self):
        return self._metric


class ExpLogger:
    def __init__(self, log_dir, exp_name, cfg=None, project_name='OpenPCDet', logger_type='tensorboard') -> None:
        self.logger_type = logger_type
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.summary_metric = dict()
        self.defined_metric = set()
        if logger_type == 'tensorboard':
            self.logger = SummaryWriter(log_dir=log_dir)
        elif logger_type == 'dummy':
            warnings.warn('!!! A dummy logger will save nothing !!!')
            self.logger = None
        else:
            raise NotImplementedError(
                f"Unsupported logger_type={logger_type!r}; use 'tensorboard' or 'dummy'."
            )

    def add_scalar(self, name, value, step):
        if isinstance(value, dict):
            self.add_scalars(name, value, step)
            return
        if self.logger_type == 'tensorboard':
            self.logger.add_scalar(name, value, step)
        elif self.logger_type == 'dummy':
            pass
        else:
            raise NotImplementedError

    def add_scalars(self, group_name, scalar_dict, step):
        if self.logger_type == 'tensorboard':
            self.logger.add_scalars(group_name, scalar_dict, step)
        elif self.logger_type == 'dummy':
            pass
        else:
            raise NotImplementedError

    def add_image(self, name, img, step, img_caption=None):
        if self.logger_type == 'tensorboard':
            self.logger.add_image(name, img, step)
        elif self.logger_type == 'dummy':
            pass
        else:
            raise NotImplementedError

    def update_summary_metric(self, name, value):
        if self.logger_type in ('tensorboard', 'dummy'):
            return
        raise NotImplementedError

    def define_metric(self, name, summary):
        if self.logger_type in ('tensorboard', 'dummy'):
            return
        raise NotImplementedError
