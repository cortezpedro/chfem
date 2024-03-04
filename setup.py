from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os, glob, subprocess
from pathlib import Path

class CustomBuildExt(build_ext):
    def run(self):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        ext = self.extensions[0]
        for source in ext.sources:
            if source.endswith(".cu"):
                compile_command = f"nvcc -Xcompiler -fPIC,-fopenmp -c {source} -odir {os.path.abspath(self.build_temp)}"
            elif source.endswith(".c") and 'wrapper' not in source:
                compile_command = f"nvcc  -Xcompiler -fPIC,-fopenmp -c {source} -odir {os.path.abspath(self.build_temp)}"
            else:  # wrapper needs Python.h
                compile_command = f"nvcc  -Xcompiler -fPIC,-fopenmp -c {source} -odir {os.path.abspath(self.build_temp)}" + \
                    ' '.join([f" -I{include_dir}" for include_dir in self.include_dirs])
            
            subprocess.check_call(compile_command, shell=True)
        ext.sources = []
        super().run()  # Continue with the regular build process
    
    def build_extensions(self):
        built_objects = glob.glob(os.path.join(self.build_temp, "*.o"))
        lib_filename = os.path.join(self.build_lib, self.ext_map['wrapper']._file_name)
        Path(os.path.split(lib_filename)[0]).mkdir(parents=True, exist_ok=True)
        linker_command = f"nvcc -Xcompiler -fopenmp -shared -o {lib_filename} {' '.join(built_objects)}"
        subprocess.check_call(linker_command, shell=True)
        super().build_extensions()

sources  = glob.glob('chfem/**/*.c', recursive=True)
sources += glob.glob('chfem/**/*.cu', recursive=True)
sources = [src for src in sources if '__pycache__' not in src]

setup(
    name='chfem',
    version='1.0',
    author="chfem team",
    url="https://gitlab.com/cortezpedro/chfem_gpu",
    description='Python API for chfem',
    packages=find_packages(),
    ext_modules=[Extension('chfem.wrapper', sources=sources)],
    cmdclass={'build_ext': CustomBuildExt}
)
