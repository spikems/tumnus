cd demo
echo "test model 6. train lr with about 1150 bigram features selected by l1 penalty"
echo
 python -m tumnus.preprocess.cut -f dump_ziran_1.txt
 python -m tumnus.learn.train --testmodel mzc_ziran_lrl1_balanced_no_count_0_1-2 --testfile dump_ziran_1.cut --appname _test33

echo 
echo "test model 7. train lr with about 850 unigram features selected by l1 compressed model"
echo
