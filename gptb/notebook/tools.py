import os


def init_notebook():
    init_notebook_file = os.path.join(os.path.dirname(__file__), 'init_notebook.py')
    get_ipython().ex('get_ipython().run_line_magic(\'run\', \'{}\')'.format(init_notebook_file))
