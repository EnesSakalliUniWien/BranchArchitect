#!/bin/bash

# Check if Java is installed
if ! java -version &> /dev/null; then
    echo "Java is not installed. Please install Java using Homebrew:"
    echo "brew install openjdk"
    exit 1
else
    echo "Java is installed."
fi

# Check if Homebrew is installed (required for Gradle installation)
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ $? -ne 0 ]; then
        echo "Failed to install Homebrew."
        exit 1
    fi
else
    echo "Homebrew is installed."
fi

# Check if Gradle is installedå
if ! gradle -v &> /dev/null; then
    echo "Gradle is not installed. Installing Gradle with Homebrew..."
    brew install gradle
    if [ $? -ne 0 ]; then
        echo "Failed to install Gradle."
        exit 1
    fi
else
    echo "Gradle is installed."
fi

# Clone the TreeCmp repository
echo "Cloning TreeCmp repository..."
git clone --recursive https://github.com/TreeCmp/TreeCmp.git

# Check if the repository was cloned
if [ ! -d "TreeCmp" ]; then
    echo "Failed to clone the TreeCmp repository."
    exit 1
fi

# Change to the TreeCmp directory
cd TreeCmp

# Build the TreeCmp project
echo "Building the TreeCmp project with Gradle..."
gradle jar

# Check if the build was successful
if [ ! -f "build/libs/TreeCmp.jar" ]; then
    echo "Failed to build the TreeCmp project."
    exit 1
else
    echo "TreeCmp built successfully."
fi

# Display final success message
echo "Installation successful. TreeCmp is ready to use."
echo "Run TreeCmp with: java -jar build/libs/TreeCmp.jar [options]"