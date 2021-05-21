import os
import inspect
import shutil
import time
import logging

import yaml

from .networks import rm_tensorboard_folder

class Experiment:
    def __init__(self, *args, log_to_file=False, allow_new=False, **kwargs):
        self._config = None
        self._folders = None
        self._logger = None

        self._log_to_file = log_to_file
        self._current_script_filename = os.path.normpath(inspect.stack(1)[1].filename)

        self._load()
        if self._config is None:
            if allow_new:
                self._new_config(*args, **kwargs)  # pylint: disable=no-member
                self._reset_folders()
            else:
                raise Exception('Could not load experiment')

    def abs_path(self, folder):
        if folder is not None:
            if folder[0] != '/':
                folder = os.path.join(self._folders['working_folder'], folder)  # pylint: disable=no-member
            folder = os.path.normpath(folder)
        return folder

    def _init_logger(self, log_filename=None):
        if log_filename is None:
            log_filename = self.abs_path(self._config.get('log_filename', 'run.log'))
        # self._logger = logging.getLogger(self._config.get('exp_name', 'general_experiment'))
        self._logger = logging.getLogger(f'exp_{time.strftime("%Y_%m_%d-%H_%M_%S")}')

        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        prefix = self._config.get('log_prefix', '')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f'-> {prefix}{{message}}', style='{'))
        handler.setLevel(logging.INFO)
        self._logger.addHandler(handler)

        if self._log_to_file:
            self._logger.info(f'Logging to "{log_filename}"')
            handler = logging.FileHandler(log_filename)
            handler.setFormatter(logging.Formatter(f'{{asctime}} | {prefix}{{message}}', style='{'))
            handler.setLevel(logging.DEBUG)
            self._logger.addHandler(handler)

    def _switch_log_file(self, log_filename=None):
        if self._log_to_file:
            if log_filename is None:
                log_filename = self.abs_path(self._config.get('log_filename', 'run.log'))

            old_handler = self._logger.handlers.pop()
            old_log_filename = old_handler.baseFilename
            if os.path.normpath(log_filename) != os.path.normpath(old_log_filename):
                self._logger.info(f'Logging to "{log_filename}"')
                if os.path.isfile(old_log_filename):
                    if os.path.isfile(log_filename):
                        with open(log_filename, 'w') as outfile:
                            with open(old_log_filename) as infile:
                                outfile.write(infile.read())
                        os.remove(old_log_filename)
                    else:
                        shutil.move(old_log_filename, log_filename)

                handler = logging.FileHandler(log_filename)
                handler.setFormatter(logging.Formatter(old_handler.formatter._fmt, style='{'))  # pylint: disable=protected-access
                handler.setLevel(logging.DEBUG)
                self._logger.addHandler(handler)

    def _load(self):
        script_folder = os.path.dirname(self._current_script_filename)
        config_filename = os.path.join(script_folder, 'config.yaml')
        if os.path.isfile(config_filename):
            config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
            self._folders = config['folders']
            self._config = config['config']
            if script_folder != os.path.normpath(self._folders['working_folder']):
                raise Exception('Trying to load experiments but the script''s folder is different then the working folder')

            self._init_logger()
            self._logger.info(f'Experiment loaded from "{script_folder}"')

    def delete(self):
        for field, folder in self._folders.items():
            if os.path.isdir(folder):
                if folder[-2:] != '__':
                    raise Exception('Experiment folder must end with double underscroe ("__") to make sure no unwanted folder is deleted')
                if field != 'tensorboard_folder':
                    self._logger.info('Removing folder: "{}"'.format(folder))
                    shutil.rmtree(folder)
                else:
                    rm_tensorboard_folder(folder, logger=self._logger)

    def _reset_folders(self):
        tmp_log_filename = f'/tmp/exp_{time.strftime("%Y_%m_%d-%H_%M_%S")}.log'
        self._init_logger(log_filename=tmp_log_filename)

        self.delete()
        os.makedirs(self._folders['working_folder'])

        shutil.copyfile(self._current_script_filename, self.abs_path('exp.py'))
        config = {'config': self._config, 'folders': self._folders}
        open(self.abs_path('config.yaml'), 'w').write(yaml.dump(config, default_flow_style=False))
        self._logger.info(f'New experiment created at "{self._folders["working_folder"]}"')
        self._switch_log_file()
