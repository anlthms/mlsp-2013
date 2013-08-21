#!/bin/bash
cat mlsp_contest_dataset2/essential_data/rec_labels_test_hidden.txt | grep -v "?" > trainlabels.txt
echo "rec_id,[labels]" > testlabels.txt
cat mlsp_contest_dataset2/essential_data/rec_labels_test_hidden.txt | grep  "?" >> testlabels.txt
python codebook.py
python birds.py
