import sys
import argparse

import json
import yaml

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.defaults_file_prefix_char = kwargs.pop('defaults_file_prefix_char', None)
        self._json_args = []
        self._list_args = {}
        self._template_args = []

        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        is_json = ('type' in kwargs) and (kwargs['type'] == 'json')
        is_list = ('type' in kwargs) and (kwargs['type'] == 'list')
        is_template = ('type' in kwargs) and (kwargs['type'] == 'template')
        if is_json or is_template or is_list:
            kwargs['type'] = str

        action = super().add_argument(*args, **kwargs)

        if is_json:
            self._json_args.append(action.dest)
        if is_list:
            self._list_args[action.dest] = kwargs.get('sep', ',')
        if is_template:
            self._template_args.append(action.dest)

        return action

    def parse_known_args(self, args=None, namespace=None):
        # types = {}
        # for action in self._actions:
        #     types[action.dest] = action.type if (action.type is not None) else str

        args_set = set(vars(super().parse_known_args()[0]).keys())

        ## Copyied from: https://github.com/python/cpython/blob/65dcc8a8dc41d3453fd6b987073a5f1b30c5c0fd/Lib/argparse.py#L1822
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        ## Based on: https://github.com/python/cpython/blob/65dcc8a8dc41d3453fd6b987073a5f1b30c5c0fd/Lib/argparse.py#L2107
        # expand arguments referencing files
        new_args = []
        for arg_string in args:

            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] != self.defaults_file_prefix_char:
                new_args.append(arg_string)

            # replace arguments referencing files with the file content
            else:
                try:
                    defaults = yaml.load(open(arg_string[1:]).read(), Loader=yaml.FullLoader)
                    for key in (set(defaults.keys()) - args_set):
                        print('!! Warning: agument "{}" might not be a valid argument'.format(key))
                    self.set_defaults(**defaults)
                except OSError:
                    err = sys.exc_info()[1]
                    self.error(str(err))

        namespace, args = super().parse_known_args(new_args, namespace)
        namespace_vars = vars(namespace)

        for key in set(self._json_args) & set(namespace_vars.keys()):
            val = namespace_vars[key]
            if isinstance(val, str):
                setattr(namespace, key, json.loads(val))

        for key in set(self._list_args.keys()) & set(namespace_vars.keys()):
            val = namespace_vars[key]
            if isinstance(val, str):
                setattr(namespace, key, val.split(self._list_args[key]))

        for key in set(self._template_args) & set(namespace_vars.keys()):
            val = namespace_vars[key]
            if not val is None:
                while True:
                    new_val = val.format(**namespace_vars)
                    if new_val == val:
                        break
                    val = new_val
                setattr(namespace, key, val)

        return namespace, args
