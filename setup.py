import setuptools

setuptools.setup(
    name="holmes",
    version="0.1",
    author="Mayalen Etcheverry",
    author_email="mayalen.etcheverry@inria.fr",
    license="MIT",
    packages=["exputils", "autodisc", "goalrepresent"],
    install_requires=[
        "neat-python==0.92",
        "pyefd",
    ],
)
