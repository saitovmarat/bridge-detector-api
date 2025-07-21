from setuptools import setup

setup(
    name="bridges_detection_api",
    version="0.1.0",
    packages=[
        'bridges_detection_api',
    ],
    package_dir={'bridges_detection_api':
                 'bridges_detection_api'},
    include_package_data=True,
    license="MIT license",
    test_suite='tests'
)
