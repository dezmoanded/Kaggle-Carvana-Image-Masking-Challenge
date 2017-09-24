from distutils.core import setup, Extension

cdecompressmodule = Extension('cdecompress',
                              sources = ['cdecompressmodule.c'])

setup(name='PackageName',
      version='1.0',
      description='Copy values to numpy data buffer.',
      ext_modules=[cdecompressmodule])