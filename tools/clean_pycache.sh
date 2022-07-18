#!/bin/bash

# Find files
files=$(find . -name '*__pycache__*' -print)

# End if no files found
if [[ -z $files ]]; then
    echo "No __pycache__'s found, terminating early"
    exit
fi

# Confirm files
echo "Will remove the following files:"
echo $files

# Proceed?
# while true; do
#     read -p "Delete the files? [y/n] " yn
#     case $yn in
#         [Yy]* ) break;;
#         [Nn]* ) exit;;
#         * ) echo "Please answer yes or no.";;
#     esac
# done

# Proceed with if/elif/else
read -p "Delete the $(wc -w <<< $files) dir(s)? [y/n] " yn
typeset -l yn
if [ ${yn::1} = "n" ]; then
    exit
elif [ ${yn::1} != "y" ]; then
    echo "Incorrect input, type [y/n]"
    exit
fi

# Remove
for file in $files
do
    rm -rf $file
    echo "Deleted:" $file 
done