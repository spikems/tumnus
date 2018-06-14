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
    echo '$1:'$1   '$2:'$2   '$3:':$3
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
banner "Step.1 dump dataset and preprocess"
if [ -f "data2.cut" ]; then
    echo "data2.cut exists, skip this step"
else

# $datadir/美赞臣*.xlsx.bz2 .
# bzip2 -d *.bz2
# python -m tumnus.preprocess.xlsdumper --dataset=jichu 美赞臣品牌及其竞品类数据-基础分类-示例.xlsx 
# python -m tumnus.preprocess.cut -f dump_jichu_0.txt
# python -m tumnus.preprocess.dedup -k3 -i dump_jichu_0.cut -o data1.cut
 python -m tumnus.preprocess.xlsdumper --dataset=ziran 美赞臣品牌及奶粉类数据（0412-0502）-自然分类.xlsx
 python -m tumnus.preprocess.cut -f dump_ziran_0.txt
 python -m tumnus.preprocess.dedup -k3 -i dump_ziran_0.cut -o data2.cut
fi
exit
#
# train/test split
#
banner "Step.2 train/test dataset split"
if [ -f splitbyday/train-05-14.cut ]; then
    echo "train/test split .cut exists, skip this step"
else

 mkdir -p splitbyday
 cd splitbyday/
#  python -m tumnus.preprocess.daysplit ../data1.cut data1
# python -m tumnus.preprocess.daysplit ../data2.cut data2
 echo "select 10 days for training dataset, another 10 days for test dataset"
# make_dataset data2-425 train-05-14.cut "30,32,33,34,35,36,37,38,39,40,41,42,43,44"
# make_dataset data2-425 test-20-29.cut "45,46,47"
 
 cat data2-4*>train-05-14.cut
 cat 7*>test-20-29.cut
 cd ..
 ln -s splitbyday/train-05-14.cut train1.cut
 ln -s splitbyday/test-20-29.cut test1.cut
 
fi
#
# train models
#
banner "Step.3 train models"
echo 
echo "model 1. train lr with all unigram features"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lr --vectorizer_type count --balanced --debug
echo 
echo "model 2. train lr with 1200 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lr --vectorizer_type count --balanced --fs_type chi2 --n_features 1200 --debug
e
echo 
echo "model 3. train lr with 500 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lr --vectorizer_type count --balanced --fs_type chi2 --n_features 500 --debug
echo 
echo "model 4. train lr with 300 unigram features selected by chi2"
echo
 python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lr --vectorizer_type count --balanced --fs_type chi2 --n_features 300 --debug
echo 
echo "model 5. train lr with about 850 unigram features selected by l1 penalty"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lrl1 --vectorizer_type count --balanced --debug 
echo 
echo "model 6. train lr with about 1150 bigram features selected by l1 penalty"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lrl1 --vectorizer_type count --balanced --ngram_range 1,2 --debug 
echo 
echo "model 7. train lr with about 850 unigram features selected by l1 compressed model"
echo
python -m tumnus.learn.train --trainfile train1.cut --testfile test1.cut --appname mzc_ziran --classifier lr --vectorizer_type count --balanced --fs_type l1 --debug
exit ()
#
# test model on new testset
#
banner "Step.4 test models on unseen dataset"
echo 
echo "test model 3. train lr with 500 unigram features selected by chi2"
echo

 python -m tumnus.learn.train --testmodel mzc_ziran_lr_balanced_chi2_count_500 --testfile splitbyday/data2-42731.cut --appname _test31

echo 
echo "test model 6. train lr with about 1150 bigram features selected by l1 penalty"
echo

 python -m tumnus.learn.train --testmodel mzc_ziran_lrl1_balanced_no_count_0_1-2 --testfile splitbyday/data2-42731.cut --appname _test31

echo 
echo "test model 7. train lr with about 850 unigram features selected by l1 compressed model"
echo

 python -m tumnus.learn.train --testmodel mzc_ziran_lr_balanced_l1_count_0 --testfile splitbyday/data2-42731.cut --appname _test31

 echo
 echo "byebye!"
 echo
