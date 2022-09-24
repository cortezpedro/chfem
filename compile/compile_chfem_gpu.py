'''
  INSTRUCTIONS:
   ~$ python (or python3) compile_chfem_gpu.py -h (or --help)
'''
#---------------------------------------------------------------------------
import sys
import os
import subprocess as terminal
import shutil
from glob import glob

# default executable name
executable_name = 'chfem_gpu'
#---------------------------------------------------------------------------
def handle_error(cmd):
  print('Command \"{}\" failed.'.format(cmd))
  print('Failed to compile {}.'.format(executable_name))
  exit()
#--------------------------------------------------------------------------- 
def printVersion():
  print('LCC - UFF')
  print('Python script to compile chfem_gpu. Version 1.1 (2020-2022)')
#---------------------------------------------------------------------------
def printHelp():
  printVersion()
  print('----------------------------------------')
  print('~$ python3 compile_chfem_gpu.py [options]')
  print('----------------------------------------')
  print('\t(-h or --help)          : print this info and exit.')
  print('\t(-v or --version)       : print version header.')
  print('\t(-n or --name) <string> : provide executable name.')
  print('\t <additional flags for the nvcc compiler> :')
  print('\t\t-DCUDAPCG_QUIET_ITERATIONS')
  print('\t\t-DCUDAPCG_VAR_32BIT')
  print('\t\t-DCUDAPCG_VAR_64BIT ------- (default)')
  print('\t\t-DCUDAPCG_MATKEY_8BIT')
  print('\t\t-DCUDAPCG_MATKEY_16BIT ---- (default)')
  print('\t\t-DCUDAPCG_MATKEY_32BIT')
  print('\t\t-DCUDAPCG_MATKEY_64BIT')
  print('\t\t-DCUDAPCG_BLOCKDIM_32')
  print('\t\t-DCUDAPCG_BLOCKDIM_64')
  print('\t\t-DCUDAPCG_BLOCKDIM_128')
  print('\t\t-DCUDAPCG_BLOCKDIM_256')
  print('\t\t-DCUDAPCG_BLOCKDIM_512 ---- (default)')
  print('\t\t-DCUDAPCG_BLOCKDIM_1024')
  print('\t\t-DELASTIC_2D_PLANESTRAIN -- (default)')
  print('\t\t-DELASTIC_2D_PLANESTRESS')
  print('\t\t-DCUDAPCG_TRACK_STOPCRIT')
#---------------------------------------------------------------------------
def checkInput():

  # check if user asked for help
  if '-h' in sys.argv or '--help' in sys.argv:
    printHelp()
    exit()
  
  # check if user asked for version
  if '-v' in sys.argv or '--version' in sys.argv:
    printVersion()
    exit()

  # check if user provided a name for executable
  exe_name = None
  input_list = sys.argv
  if '-n' in sys.argv or '--name' in sys.argv:
    try:
      flag_id = input_list.index('-n')
    except ValueError:
      flag_id = input_list.index('--name')
    flag = input_list.pop(flag_id)
    if (flag_id >= len(input_list)):
      print('ERROR: flag \"{}\" must be followed by a string with a name for the executable.'.format(flag))
      print('Aborted before compiling.')
      exit()
    exe_name = input_list.pop(flag_id)

  # append the rest of the input strings as one long string
  string = ''
  for i in range(1,len(input_list)):
    string += ' '+input_list[i]
  
  return string, exe_name
#---------------------------------------------------------------------------
def extractFilename(filepath):
  _, filename = os.path.split(filepath)
  return os.path.splitext(filename)[0]
#---------------------------------------------------------------------------

#--------------
#---- main ----
#--------------

if __name__ == '__main__':

  # check OS
  if sys.platform.lower() == 'darwin': # mac
    print('ERROR: This script is not meant for macOS.')
    exit()

  isLinux = ('linux' in sys.platform.lower())

  # check input
  input_string, exe_string = checkInput();
  if not (exe_string is None):
    executable_name = exe_string

  executable_name += '' if isLinux else '.exe'
     
  # get names of files with .cu and .c extensions and store
  # in list with .o extension (will be used for linking)
  o_ext = '.o' if isLinux else '.obj'
  o_files  = [extractFilename(f)+o_ext for f in glob("../src/**/*.cu",recursive=True)]
  o_files += [extractFilename(f)+o_ext for f in glob("../src/**/*.c", recursive=True)]
  
  # generate one string containing all elements in list
  o_files_string = ''
  for f in o_files:
    o_files_string += ' '+f

  # array of compiler directives
  extra_compiler_options = '  --compiler-options -std=gnu99' if isLinux else ''
  if isLinux:
    compiler_directives = [
      'nvcc -c ../src/femhmg/cudapcg/kernels/*.cu'+input_string,
      'nvcc -c ../src/femhmg/cudapcg/solvers/*.cu'+input_string,
      'nvcc -c ../src/femhmg/cudapcg/*.cu'+input_string,
      'nvcc -c ../src/femhmg/physical_phenomena/*.c'+input_string + extra_compiler_options,
      'nvcc -c ../src/femhmg/report/*.c'+input_string + extra_compiler_options,
      'nvcc -c ../src/femhmg/*.c'+input_string + extra_compiler_options,
      'nvcc -c ../src/main.c'+input_string + extra_compiler_options,
      'nvcc -o {}{} -Xcompiler -fopenmp -O3'.format(executable_name,o_files_string)+input_string
    ]
  else: # windows
    cuda_files = [f.replace('\\\\','\\').replace('/','\\') for f in glob("../src/**/*.cu",recursive=True)]
    c_files = [f.replace('\\\\','\\').replace('/','\\') for f in glob("../src/**/*.c",recursive=True)]
    compiler_directives = []
    for f in cuda_files:
      if '\\kernels' in f:
        compiler_directives.append('nvcc -c '+f+input_string)
    for f in cuda_files:
      if '\\solvers' in f:
        compiler_directives.append('nvcc -c '+f+input_string)
    for f in cuda_files:
      if not('\\kernels' in f) and not('\\solvers' in f):
        compiler_directives.append('nvcc -c '+f+input_string)
    for f in c_files:
      if '\\physical_phenomena' in f:
        compiler_directives.append('nvcc -c '+f+input_string+extra_compiler_options)
    for f in c_files:
      if '\\report' in f:
        compiler_directives.append('nvcc -c '+f+input_string+extra_compiler_options)
    for f in c_files:
      if not('\\physical_phenomena' in f) and not('\\report' in f) and not('main' in f):
        compiler_directives.append('nvcc -c '+f+input_string+extra_compiler_options)
    compiler_directives.append('nvcc -c ..\src\main.c'+input_string+extra_compiler_options)
    compiler_directives.append('nvcc -o {}{} -Xcompiler \"-openmp\" -O3'.format(executable_name,o_files_string)+input_string)

  # run compiler commands in terminal
  for command in compiler_directives:
    if isLinux:
      print(command)
    if (terminal.call(command,shell=True)):
      handle_error(command)
  
  if isLinux:
    terminal.call('rm {}'.format(o_files_string),shell=True)
    shutil.move('{}'.format(executable_name), '../{}'.format(executable_name))
  else:
    terminal.call('del {}'.format(o_files_string),shell=True)
    shutil.move('{}'.format(executable_name), '../{}'.format(executable_name))
    shutil.move(extractFilename(executable_name)+'.exp', '../{}'.format(extractFilename(executable_name)+'.exp'))
    shutil.move(extractFilename(executable_name)+'.lib', '../{}'.format(extractFilename(executable_name)+'.lib'))

  
  # print feedback that compiling went ok  
  print('{} binary executable successfully compiled.'.format(executable_name))
#---------------------------------------------------------------------------

