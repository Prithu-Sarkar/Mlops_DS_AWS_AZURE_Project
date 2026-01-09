from setuptools import find_packages, setup  # Import setuptools utilities for package discovery and setup configuration
from typing import List  # Type hint for function return type


HYPEN_E_DOT = '-e .'  # Constant representing editable install flag from requirements.txt (pip install -e .)
                                              # This flag is removed during packaged installation as it's for development only

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt, cleans newline characters from each line,
    removes the editable install flag ('-e .') if present, and returns 
    the list of clean dependency names for setup() install_requires.
    
    Args:
        file_path (str): Path to requirements.txt file
    
    Returns:
        List[str]: Cleaned list of package requirements
    """
    requirements = []  # Initialize empty list to store requirements
    
    with open(file_path) as file_obj:  # Open requirements.txt file safely (auto-closes)
        requirements = file_obj.readlines()  # Read all lines into list
        requirements = [req.replace("\n", "") for req in requirements]  # Strip newlines from each requirement
        
        # Remove editable install flag if present (not needed for packaged distribution)
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements  # Return cleaned requirements list


# Configure and register the ML project package for pip installation
setup(
    name='Mlproject',                    # Package name on PyPI
    version='0.0.1',                     # Semantic version (MAJOR.MINOR.PATCH)
    author='Prithu',                      # Author name
    author_email='prithusarkar90@gmail.com', # Contact email for PyPI
    packages=find_packages(),            # Auto-discover all Python packages in project
    install_requires=get_requirements('requirements.txt')  # Dynamically load dependencies from requirements.txt
)
