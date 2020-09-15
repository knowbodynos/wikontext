from setuptools import setup

def get_version():
    with open("VERSION", 'r') as fh:
        return fh.read().strip('\n')

setup(
    name='wikontext',
    version=get_version(),
    packages=['wikontext'],
    url='https://github.com/knowbodynos/wikontext',
    license='',
    author='Ross Altman',
    author_email='ross@rossealtman.com',
    description='Smarter page previews for a smoother Wikipedia experience'
)
