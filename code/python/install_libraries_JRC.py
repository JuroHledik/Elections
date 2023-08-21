import os
import sys

# Get the working directory
path = os.getcwd()

# Append the /code directory to use the user-defined scripts within
sys.path.append(path + '/code')

# Import the user-defined functions (this might be grayed out in PyCharm, ignore if so)
import config

os.system("pip install --proxy http://"+config.proxy_username+":"+config.proxy_password+"@ps-isp-usr.cec.eu.int:8012  plotly")


# Add the package location to the path:
# SPYDER -> Tools -> PYTHONPATH manager -> Add path:
# C:\Users\hlediju\AppData\Roaming\Python\Python39\site-packages

# This runs MATLAB in the command prompt:
# C:\PsTools\PsExec.exe -u isis\iazzusr -p ecf1n_9004 -W C:\Users\Public\MATLAB "C:\Program Files\MATLAB\R2021b\bin\matlab.exe" -nodisplay -nosplash -nodesktop -r "cd \\s-jrcipsc01-cifs.jrc.it\FEA\JHledik\Symbol_v2\matlab_code; run('ScriptSymbolDatabase.m'); exit;"