from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific because ctypes shared libraries are bundled."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
