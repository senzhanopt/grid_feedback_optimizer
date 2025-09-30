from setuptools import setup, find_packages

setup(
    name="grid_feedback_optimizer",
    version="0.1.0",
    packages=find_packages(where="src"),  # <-- look inside src/
    package_dir={"": "src"},              # <-- root of packages is src/
    install_requires=[
    ],
    python_requires="==3.11.*",
)