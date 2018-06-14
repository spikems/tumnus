#!/bin/bash
if [ -z "$_tumnusproject_" ] 
then
    echo "tumnus env not set yet, quit"
    echo "run \"source ~/wangwei/tumnus/bin/init_env.sh\" first"
    source ~/wangwei/tumnus/bin/init_env.sh
    exit 1
else
    echo "tumnus env is already set"
fi

make_dataset()
{
    prefix=$1
    outname=$2
    days=$3
    echo $prefix  $ooutname $days   
    echo "make dataset..."
    for day in $days; do
	echo $days
        cat $prefix$day.cut >>$outname
    done
    
    #validate
    echo "days in $outname are:"
    gawk '{print $2}' $outname |sort | uniq
     

}

banner()
{
    echo "==============="
    echo $1
    echo "==============="
}

datadir=$_tumnusproject_/data

echo $datadir

mkdir -p demo
echo `pwd`
cd demo

#
# dump dataset
#

banner "Step.1 dump dataset and preprocess"
if [-f "data1.cut" ]
then
    echo "data1.cut exists, skip this step"
else
    python -m tumnus.preprocess.xlsdumper1 --dataset=xiaomi xiaomi.xlsx

