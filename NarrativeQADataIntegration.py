# -*- coding: utf-8 -*-

import os, sys
import json
import re
import csv
from collections import OrderedDict


def read_irregular_verblist(fname_irregular_verblist):
    fpIn = open(fname_irregular_verblist, 'rt', encoding='utf8')
    csvreader = csv.reader(fpIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    _ = next(csvreader)
    dict_irregular_verblist = dict()
    for arow in csvreader:
        if arow[2].find('/') != -1:
            dict_irregular_verblist[arow[0]] = arow[2].split('/')[0]
        else:
            dict_irregular_verblist[arow[0]] = arow[2]
    return dict_irregular_verblist

def read_en_verbs_without_be(fname_en_verbs_without_be):
    fpIn = open(fname_en_verbs_without_be, 'rt', encoding='utf8')
    # 23 columns
    csvreader = csv.reader(fpIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    # _ = next(csvreader)
    # 23
    set_words = set()
    for arow in csvreader:
        for aword in arow:
            if aword != '':
                set_words.add(aword)
    fpIn.close()
    return set_words


if __name__ == '__main__':
    set_en_verbs_without_be = read_en_verbs_without_be('./en_verbs_without_be.csv')
    # print(set_en_verbs_without_be)
    dct_irregular_verblist = read_irregular_verblist('./irregular_verb_list.txt')
    # print(dct_irregular_verblist)
    dct_reverse_irregular_verblist = {k: v for v, k in dct_irregular_verblist.items() if v != '...'}   
    lst_irregular_verblist = list(dct_reverse_irregular_verblist.keys())
    str_joined_irregular_verblist = '|'.join(lst_irregular_verblist)
    fname_annotations = './narrativeqa/qaps.csv'
    fpIn_annotations = open(fname_annotations, 'rt', encoding='utf8')
    csvreader = csv.reader(fpIn_annotations, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    # pattern_string_match_avoid = '^who (was|is) (\w+ing) .+$'
    # 01aa10d75658840a478ede17631dba875651c370
    pattern_string_match_avoid = f'^who (was|is) (\w+ing|{str_joined_irregular_verblist}|\w+ed) .+$'
    pattern_string_match_accept = '^who (was|is) .+$|^what is the name of .+$|^what is .+ name\?$'

    pt_avoid = re.compile(pattern_string_match_avoid)
    pt_accept = re.compile(pattern_string_match_accept)

    fnameoutInt = './narrativeqaIntegrated.csv'
    fpOut_Int = open(fnameoutInt, 'wt', encoding='utf8')
    csvwriter = csv.writer(fpOut_Int, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    row0 = next(csvreader)
    csvwriter.writerow(row0)
    for arow in csvreader:
        # document_id,set,question,answer1,answer2,question_tokenized,answer1_tokenized,answer2_tokenized
        is_verb_in_question = False
        for aword in arow[5].lower().split(' '):
            if aword in set_en_verbs_without_be:
                is_verb_in_question = True
                break
        if is_verb_in_question:
            continue
        if arow[5].lower().find('protagonist') != -1 or arow[5].lower().find('main character') != -1 or arow[5].lower().find('antagonist') != -1:
            continue
        if pt_avoid.match(arow[5].lower()) is not None:
            continue
        if pt_accept.match(arow[5].lower()) is not None:
            arow_result = arow
            csvwriter.writerow(arow_result)
        
    fpOut_Int.close()
    fpIn_annotations.close()
