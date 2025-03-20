# Base directory to search for packages
base_dir="src"

# Find all directories under src that contain a setup.py or pyproject.toml
package_dirs=$(find "$base_dir" -type f \( -name "setup.py" -o -name "pyproject.toml" \) -exec dirname {} \; | sort -u)

# Iterate through each package directory and perform an editable install
for dir in $package_dirs; do
  echo "Performing editable install for package in $dir..."
  pip install -e "$dir"
  if [[ $? -ne 0 ]]; then
    echo "Error installing package in $dir. Exiting."
    exit 1
  fi
done
