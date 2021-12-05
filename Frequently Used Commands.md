# Frequently Used Commands
Zhenhong Hu

Last Updated: 12/04/2021

---
## Virtual Environment
### Create the virtual environment
To create a virtual environment in a given directory, type:
```python
python -m venv /path/to/directory
```
### Activate the virtual environment
- On Unix or MacOS, using the bash shell: 
``` source /path/to/venv/bin/activate ```
- On Windows using the Command Prompt: 
``` path\to\venv\Scripts\activate.bat ```
- On Windows using PowerShell: 
``` path\to\venv\Scripts\Activate.ps1 ```

Activating the virtual environment will change your shell’s prompt to show what virtual environment you’re using, and modify the environment so that running python will get you that particular version and installation of Python. For example:
``` python 
path\to\venv\Scripts\Activate.ps1
(quantvenv) PS C:\Users\project>
```


## Pip Commands
- To display all of the packages installed in the virtual environment:
``` pip list ```
- To upgrade the package to the latest version:
``` python -m pip install --upgrade package_name ```
- To display information about a particular package:
``` pip show package_name ```
- pip freeze will produce a similar list of the installed packages, but the output uses the format that pip install expects. A common convention is to put this list in a requirements.txt file:
``` pip freeze > requirements.txt ```
- The requirements.txt can then be committed to version control and shipped as part of an application. Users can then install all the necessary packages with install -r:
``` python -m pip install -r requirements.txt ```
- To remove the packages from the virtual environment:
``` pip uninstall package_name ```


``` xxxxxx ```







