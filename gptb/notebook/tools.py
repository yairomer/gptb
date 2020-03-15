import os


def init_notebook(jupyter_frontend=True):
    init_notebook_file = os.path.join(os.path.dirname(__file__), 'init_notebook.py')
    get_ipython().ex('get_ipython().run_line_magic(\'run\', \'{} {}\')'.format(init_notebook_file, int(jupyter_frontend)))


def change_theme(jupyter_frontend=True):
    if jupyter_frontend:
        from IPython.display import display, HTML
        theme_file = os.path.join(os.path.dirname(__file__), 'custom.css')
        display(HTML('<style>\n{}\n</style>'.format(open(theme_file).read())))