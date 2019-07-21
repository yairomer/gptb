import os
from setuptools import setup, find_packages

requirements_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')

# requirements_list = [str(req.req) for req in pip.req.parse_requirements(requirements_file, session=pip.download.PipSession())]
requirements_list = open(requirements_file, 'r').read().split('\n')
requirements_list = [x.replace(' ', '') for x in requirements_list]
requirements_list = [x for x in requirements_list if len(x)]

setup(name='gptb',
      version='0.1',
      description='General Purpose Tool Box',
      # url='',
      # author=''
      # author_email='',
      # license='',
      packages=find_packages(),
      install_requires=requirements_list,
      )
