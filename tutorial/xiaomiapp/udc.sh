#!/bin/bash

if [ -z "$_tumnusproject_" ] ; then
    echo "tumnus env not set yet, quit"
    echo "run \"source tumnus/bin/init_env.sh\" first"
    exit 1
fi

make_dataset()
{
    prefix=$1
    outname=$2
    days=$3
    
    echo "make dataset..."
    for day in $days; do
        cat $prefix$day.cut >>$outname
    done
    
    #validate
    echo "days in $outname are:"
    gawk '{print $2}' $outname |sort | uniq 

}

banner()
{
    echo "====================================="
    echo $1
    echo "====================================="
}

datadir=$_tumnusproject_/data
#
# this is a demo of training process
#
mkdir -p demo
cd demo

#
# dump dataset
#
#python -m tumnus.preprocess.udc_cut -f refine.udc
#python -m tumnus.preprocess.udc_cut -f udc4.txt

#exit
#
# train/test split
#
#
# train models
#
banner "Step.3 train models"
echo 
echo "model 1. train lr with all unigram features"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lr --vectorizer_type count --vocabulary feature.dict --balanced --debug
echo 
echo "model 2. train lr with 1200 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lr --vectorizer_type count --balanced  --debug --vocabulary feature.dict

echo 
echo "model 3. train lr with 500 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lr --vectorizer_type count --balanced  --debug --vocabulary feature.dict
echo 
echo "model 4. train lr with 300 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lr --vectorizer_type count --balanced  --debug --vocabulary feature.dict
echo 
echo "model 5. train lr with about 850 unigram features selected by l1 penalty"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lrl1 --vectorizer_type count --balanced --debug   --vocabulary feature.dict
echo 
echo "model 6. train lr with about 1150 bigram features selected by l1 penalty"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lrl1 --vectorizer_type count --balanced --debug --vocabulary feature.dict
echo 
echo "model 7. train lr with about 850 unigram features selected by l1 compressed model"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname xiaomi --classifier lr --vectorizer_type count --balanced --debug  --vocabulary feature.dict

#
# test model on new testset
#
exit
banner "Step.4 test models on unseen dataset"
echo
echo "test model 6. train lr with 500 unigram features selected by chi2"
echo

 python -m tumnus.learn.train --testmodel xiaomi_lrl1_balanced_chi2_count_500 --testfile splitbyday/data2-42731.cut --appname _test31

echo 
echo "test model 6. train lr with about 1150 bigram features selected by l1 penalty"
echo

 python -m tumnus.learn.train --testmodel xiaomi_lrl1_balanced_no_count_0_1-2 --testfile splitbyday/data2-42731.cut --appname _test31

echo 
echo "test model 7. train lr with about 850 unigram features selected by l1 compressed model"
echo

 python -m tumnus.learn.train --testmodel xiaomi_lr_balanced_l1_count_0 --testfile splitbyday/data2-42731.cut --appname _test31

 echo
 echo "byebye!"
 echo
