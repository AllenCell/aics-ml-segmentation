from setuptools import setup, find_packages


PACKAGE_NAME = 'aicsmlsegment'

"""
Notes:
We get the constants MODULE_VERSION from
See (3) in following link to read about versions from a single source
https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
"""

MODULE_VERSION = ""
exec(open(PACKAGE_NAME + "/version.py").read())


def readme():
    with open('README.md') as f:
        return f.read()


test_deps = ['pytest', 'pytest-cov']
lint_deps = ['flake8']
all_deps = [*test_deps, *lint_deps]
extras = {
    'test_group': test_deps,
    'lint_group': lint_deps,
    'all': all_deps
}

setup(name=PACKAGE_NAME,
      version=MODULE_VERSION,
      description='Scripts for ML structure segmentation.',
      long_description=readme(),
      author='AICS',
      author_email='jianxuc@alleninstitute.org',
      license='Allen Institute Software License',
      packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
      entry_points={
          "console_scripts": [
            "dl_train={}.bin.train:main".format(PACKAGE_NAME),
            "dl_predict={}.bin.predict:main".format(PACKAGE_NAME),
            "curator_merging={}.bin.curator.curator_merging:main".format(PACKAGE_NAME),
            "curator_sorting={}.bin.curator.curator_sorting:main".format(PACKAGE_NAME),
            "curator_takeall={}.bin.curator.curator_takeall:main".format(PACKAGE_NAME),
          ]
      },
      install_requires=[
          'numpy>=1.15.1',
          'scipy>=1.1.0',
          'scikit-image>=0.14.0',
          'pandas>=0.23.4',
          'aicsimageio==0.6.4',
          'aicsimageprocessing',
          'tqdm',
          'pyyaml',
          #'tensorboardX'
          #'pytorch=1.0.0'
      ],

      # For test setup. This will allow JUnit XML output for Jenkins
      setup_requires=['pytest-runner'],
      tests_require=test_deps,

      extras_require=extras,
      zip_safe=False
      )
