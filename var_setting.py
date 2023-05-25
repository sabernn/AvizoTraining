

import os, sys
platform_to_exec = {'win32': 'pythonw.exe', 'linux': 'bin/python3.6'}
sys.executable = os.path.abspath(os.path.join(sys.exec_prefix, platform_to_exec[sys.platform]))



