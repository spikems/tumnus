#/bin/bash
set -x
if [ -f "test.txt" ] 
then
	echo "exist"
else
	echo "not exist"
fi
