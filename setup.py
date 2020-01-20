from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='dphmix',
      version='0.2.0',
      description='Unsupervised and Semi-supervised Dirichlet Process Heterogeneous Mixtures',
      long_description='Implements Dirichlet Process Heterogeneous Mixtures of exponential family distributions for clustering heterogeneous data without choosing the number of clusters. Inference can be performed with Gibbs sampling or coordinate ascent mean-field variational inference. For semi-supervised learning, Gibbs sampling supports must-link and cannot-link constraints. A novel variational inference algorithm was derived to handle must-link constraints.',
      classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      url='https://github.com/tahmidmehdi/dphmix',
      download_url='https://github.com/tahmidmehdi/dphmix/archive/v0.2.0.tar.gz',
      author='Tahmid Mehdi',
      author_email='tfmehdi@cs.toronto.edu',
      license='GNU GPL v3.0',
      packages=['dphmix'],
      install_requires=[
          'numpy',
          'joblib',
          'scipy',
          'pandas',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)


