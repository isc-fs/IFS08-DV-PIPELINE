import os
import sys

# Get the absolute path of the parent directory of the subpackage
import os
import sys

# Get the directory of the current file (this is the top-level package)
package_dir = os.path.dirname(__file__)

# Add the package directory to sys.path if it's not already there
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# import path_planning.fsd_path_planning as fsd_path_planning
