# -*- coding: utf-8 -*-

import os, sys
import json
import re
import csv
from collections import OrderedDict
import spacy

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

def read_necessary_fnames():
    fpIn = open('./narrativeqaIntegrated_for_necessary_fnames.csv', 'rt', encoding='utf8')
    csvreader = csv.reader(fpIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    _ = next(csvreader)
    set_necessary_fnames = set()
    for arow in csvreader:
        set_necessary_fnames.add(arow[0])
    return set_necessary_fnames

def read_literatures(rootdir):
    set_necessary_fnames = read_necessary_fnames()
    # this is for debugging to obtain set_necessary_fnames_not_utf8
    set_included_in_set_necessary_fnames_and_is_NOT_utf8 = set()
    dct_literatures = dict()
    set_fnames_to_skip = {'5ab44bc45dd55c24d59eccbba8f0d52c4b651016', '6cce85cdf00f2e164a8c9fcd71467d9c46559c20', '935775d945bc210b0928bffb6924e06e1ef9a9dd', 'fb663b1df2e632d1f5362f61e4236acab8fadaf4', '399f3571af53d150a999da5553de856ddb0815b9', '5ea4ebbc0d86c3f629932a6f2470949776e58579', '5c201758473664dfb3b0cb71754ab3dbbfee35df', '1995b44b685baf07ae17d56bb810ab8295785010', '2d3b9c6a996dbf0372ad1242107bf0e2f4a845d7', '6a2f59a81c3730b5e1d5f685bac8bbbe547b7b3b', '336e272847df32ee6f87eca63ee800c5add03639', 'd9df8732f4fad8d4ffa6d8b2f7af12ea374a5be2'}
    set_fnames_to_change_read_encoding = {'a77f52738cbd7ce7658e3fe5853068eb8fc095c4', 'ad63a9aa1afa22bf51451da2e2f45d9543c9ca62', '828c99d509e808fedfb0675781a5d137c4dfe0b7', '076acafe2eb72ccf6e894bbb4b8b318f4cf3c58d', '7f1342389643da8afa5cdd6a7b512546e6f4b4c2', '681778326e846ea9979ff3aed4ee9668abe7d489'}
    set_necessary_fnames_not_utf8 = {'5c1d04428ffd3f0ddd732a723e61371e0255aa49', '266f5b2295980ad31b5090a6c51b69055c87b3a7', '64de615fe8d7b550c3f9ce72e55bab8ebac69b2f', 'cf2f6cbfb5bea59e783846202d9040713a79890f', '061c5dd1dc117e3161f1f0a7794b807932ef4675', '6aa4c6be53eff0024e5c84f99ac94cddff4eb8f0', 'a69fee0515cdb067fec5c42fa88ace5c9639118c', '36ce1aa16abd65ad7265a9601631209bb4f0c347', '1508452704b5829941b98aeee2eeba192f5beae5', '1da161b6007d46d85eda16ae391e1e78218fabd2', 'ffcf7daee9cda766d2fcf1f6399b29be41876b21', '949f179ab9de0a9bbb52fce8fecf52bf67caaccd', '214cb5277750c1ccddec0b10fa545c3c78c45f64', '84248647b95fced93f041f931426523b7d25d226', 'b1b2826a2726fe62ceae88b01bf0be24d95a9d39', 'c231d777e69681b32aff6a29bcb70932bc60c0a9', '573cdbc4363b1dfe9e2ad983d062dc036b20544f', '84732f85b51dfbfed6c40f2bc1e35e1697eade8e', '17272570cdcbfce7fda86ae35332b1fcae569c30', '9ebb84bdc9cc6d698ccc331437bd1ec3b5f0dddb', 'e3793c8072528b1d92b8614113e6d2b5748652cd', '127e1efe32b11e606a0c8f49a2399abb4a52f9d9', '2caed8cd33002af756cef1c108986384a0b7066b', '10910853a653d0e6ef540a1a3cc63ef2746a798c', '4bb855bbb2a0da63ac5717c87aba8883b829201e', '6031da6fa93ad9cac1b6da6586010aab81c7b4da', '411d53d3c5c42990aebb7cb0bf4964f8d0a6f0bc', '916835cb4bcb3baa6333e7cca25bef7710dbdcbc', 'd5615c44b01f98de34d10baaccc981995a43864d', '0434145d9284423a36a714fc55246ed0bdc39a82', 'b833410a8ff29952ca664319cb3462f2fd07d4f9', '46a5ee8bbf57f56ad5472e0712e98370b734145d', 'de01eef0c46c744cbc407f886e97195c15d133dc', '042bb7019f583cebe3290795149fdc70412ad813', '6d2e1b95ed2a00e16e046b6f3d0b03e687c7f7f7', '599b6ffc75b39b2d8dde708ad37b08dde52fccc0', '0bc7352d6a0e678c0d8acc57c0c1cc3466fe9ef7', '361364f57460139410c4130a1e7a58caf152c2bd', '2dd23dbf75e37c1f35eb8e2e317a36b7033495d0', 'c002791c4ff779710ca7ba8d8cde2ac4b27d28b3', 'ad13b1236bd9c3b925d27a487959205d209ca361', 'a87d47ee243b5cc4f8ce3acf483ccf9e77083e16', 'b08882a8306a6a4b1f405827ee19b4aca538c023', '050f88f6c8fed44ecbd7658b0f450fb705ce368d', '69099d7d543fad22f61d8acf97c681c9c86cac0e', '265fa34cf1f2c6145cce7ee9402bb3af6d898624', '80f03a1fcfa33d7dbbe32a6252022e9da17847b9', 'f7bf427e41af53409d7907160f7908e723b78eb0', '2132babdf6d70933760a9d8e9c6ac5c3305ed253', '0bec3c185ffe7f716108fce8bd4a1558d4cd4a54', '2e3b266eda62694b15254616844a26d972f5939b', 'f3dfd74b81d1cc1db840db8bceabc4da2bdf2953', 'f2454588fd5606c7a4dedfeddfbfe1cdb580754b', '5ad9844f125d7051ba23edbb8b48a74f4f6102c8', 'a8549480950ac906c9426b7d8cb7963e52e4cd6c', '10d52d13b492d0e2319b8b5d3349ce4f9eaf26d4', '405df1ccf0409ea6040c2a765a0315878f991d79', '10c992498cf8bea350d6209b4b8f1d0437440694', 'd8844d709aa624a5ffe70f185dc68488839d37ea', 'c504d1c2be689fe79324ef90f3657dc02ef2acfb', 'ab78e645d43e4f56ca208ef590f0a4662b4d8efb', 'ccdf8c8c07e95675fae3591714061ecacfd5ad2e', '39f11774cd462fe3c40283af7c3778d18b78acb9', '2b0d41cec61dc3b7faf7a011051cdbcacb55050b', 'f5255dcda3e92492cca0b95687bf01d0908b07b4', '2527b96b0a25841507a88966736c4e40181db6c9', 'ee6f2d470966fa272e00d6ad19d12a706545faff', 'a03d6a5ddf48beb93529c87571fad8e02d17b373', '70794150f324949ca49f182db0d3f8d69d0c779e', 'bb4c45e6a2f4c58def38dbd630d7059028c27bdd', '166e4604f74997601e16793ca5af0f54bc2cc81c', 'b856de66b8fd259ac0a3e5bef124efb447fe4f75', 'd8aeeba694332530d1ca1647779c3228959aa20a', '194fd3c13f34b5a640280e7f43e59dbed66c215c', '31ce5b74112c35db9004ecd486c543e9373f5d53', 'bfd50b1cf73709308ab4ad727d829c5fca23480c', '6bd5faba6a0bd427374f7e20d3044fbf09b5d1a8', 'e292eae2486862f6df6ff388cb2dd6777bc73f27', '15e5cdd93340ecf4ab97248e15f3870eb26bf10b', '3634471ed994ee7d4f382d8e7edbc56de5c28c42', '55d1940b8c8e1c73e175bac2fce0a4f9844fff02', '1f8e03c7b6a6864108933fba1906455f78e4cfa6', '0cb3433c5ac030dba47414e2655c3c49e4f37527', '9cd25b973d253386eaebbb7f2f7821dc7518f6d6', 'c085c72c78f6ad35854654d222a54b016b08dc65', '754184dec0d686d0b8b9ad7e1dc15bf0eb5027f5', 'de42ec88ce00d0f1e21ad4ad719d1d64499e5166', '9c97f8fe3e6e78aa8d35e3c4759a4d1f766000bc', '6ffb5d981d101fbbd43d97bc45720388570f2c61', 'c9fb048c6f4d00d15a69c9049c4826cc66118bb3', '570e10517cfe77fcda4a810650311dd79f3bcedc', '87c046f7ad020a2da8302de2d39c918d5c3d46d6', 'ce8cb184a11535e7a7c824c82b7772a1c3a7c92c', '00f9dbb0a851bc6099d5216e5fa8719b2ac3b82b', 'c29a505a7f2efe48bc73da66d4608888b8dc617b', '48266045f2dbe4de0cea552d3ec8ffb541c5e182', 'ec5123faf9944ed8ef012cce89123db124242b3d', '2114ee4681976de8a9d75d0e411ccc47bfa1caa3', '5c8b96c00b579c1e89e0ffeedb47e6d122e4c159', '4f17e590323c45bf12f789e5990d6ab90698671a', '9cbe9d08ff6673e8dba308ac11ba88d71b425209', '150683ab8156bdc78ebd23a2ed7f7e265b780bd0', '99585e4b51fa1bf94db82b0551d1d664e584621d', '46cf28a716c255263334b3baca9fc759c44d8766', '33563aed8b26142ec99e57b67fe18f21f7c1c794', 'ba09fb16effa053774cf76e1e27e3d46a45de048', '567c1a39aff9590875a843a9df35f8c5b880ae27', '1528f4c003dde308ae74bdec458466765944ca6a', '141a5885bb6d8df56dadbd617b5dec7db102701d', '625ee9d399b8fa5b0921bb3263a5547872ca4f77', 'f75afa70c82c3f894abccad514688a835e45c600', '728fb0c9da98fa7853adac4934987df261d51041', 'c85ec67ceff73b596af55bbd3cd88bfd37f622ec', 'e4081796caf2b0354f1fc78626b7a74396979e5b', '68ee401e0c66832834f605b625d5062b06a59515', 'febd3002298e75a9e9b5569500989766137608f8', 'ffa719867c77cfd8fc661fdb6d8c8d266746e15f', '9f83c8e49f5a53b211caf37cbdc659f97d2ef30a', '6bec7c2bdd0b01296bc9020288d833709e54cd52', '83a1fd492021ceb110451d65888072daf64a5d4f', 'd5b32abd0fe5966b8c619084932c5d832d51f063', 'd81debbae25eebaf6ac998a0059eaa08cc7be5bc', '20f1cc76a6b149b417e0d9943c7f2caac3a04875', '2b5d19eb7767f089cf6b5cee04106b964962eb00', 'ceb4f09a3b51a02fdce4260a79a139723fd541a1', 'dfc26addcd9cd2cf53ed6804445d8bc60668d316', '9e3a7d4c30143d737b783df94e7c1c2fd6fe6514', '9a50bcf24dea0f3934df9a55f6e82df03c19c7cb', '8b9da16420edd653ae5e0e2925dd3cade3a21ebc'}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.content'):
                fname_without_extension = os.path.splitext(file)[0]
                if fname_without_extension in set_fnames_to_skip:
                    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa6 in position 59297: invalid start byte
                    continue
                if fname_without_extension not in set_necessary_fnames:
                    continue
                # ceb4f09a3b51a02fdce4260a79a139723fd541a1
                # print(fname_without_extension)
                if fname_without_extension in set_fnames_to_change_read_encoding or fname_without_extension in set_necessary_fnames_not_utf8:
                    with open(os.path.join(subdir, file), 'rt', encoding='unicode-escape') as fpIn:
                        # # ISO 8859-1 is the ISO standard Latin-1 character set and encoding format. CP1252 is what Microsoft defined as the superset of ISO 8859-1. see https://stackoverflow.com/questions/19699367/for-line-in-results-in-unicodedecodeerror-utf-8-codec-cant-decode-byte
                        # # for i in range(1024):
                        # #     try:
                        # #         _ = fpIn.readline()
                        # #     except UnicodeDecodeError:
                        # #         pass
                        # # dct_literatures[fname_without_extension] = fpIn.read()
                        # if fname_without_extension == 'ceb4f09a3b51a02fdce4260a79a139723fd541a1':
                        #     # for aline in fpIn:
                        #     #     print(aline)
                        #     continue
                        # else:
                        #     dct_literatures[fname_without_extension] = fpIn.read()
                        # # .encode('unicode-escape').decode('utf-8')
                        dct_literatures[fname_without_extension] = fpIn.read()
                else:
                    # if fname_without_extension in set_necessary_fnames:
                    #     # with open(os.path.join(subdir, file), 'rt', encoding='utf-8') as fpIn:
                    #     #     dct_literatures[fname_without_extension] = fpIn.read()
                    #     try:
                    #         with open(os.path.join(subdir, file), 'rt', encoding='utf-8') as fpIn:
                    #             dct_literatures[fname_without_extension] = fpIn.read()
                    #     except UnicodeDecodeError:
                    #         # set_included_in_set_necessary_fnames_and_is_NOT_utf8.add(fname_without_extension)
                    with open(os.path.join(subdir, file), 'rt', encoding='utf-8') as fpIn:
                        dct_literatures[fname_without_extension] = fpIn.read()
    # print(set_included_in_set_necessary_fnames_and_is_NOT_utf8)
    print('len(set_necessary_fnames_not_utf8)', len(set_necessary_fnames_not_utf8))
    print('len(set_necessary_fnames)',len(set_necessary_fnames))
    return dct_literatures


def find_global_pos(the_literature, the_span_text_from, the_line_num_from, the_span_text_to, the_line_num_to):
    curr_line_beginning_pos = 0
    for line_idx, aline in enumerate(the_literature.split('\n')):
        curr_line_num = line_idx + 1        
        if curr_line_num == the_line_num_from:
            line_position = aline.find(the_span_text_from)
            if line_position == -1:
                sys.exit(f'the_span_text not found in the line {the_span_text_from},{the_line_num_from}')
            global_start_from = curr_line_beginning_pos + line_position
            global_end_from = global_start_from + len(the_span_text_from)
            assert the_literature[global_start_from:global_end_from] == the_span_text_from
        if curr_line_num == the_line_num_to:
            line_position = aline.find(the_span_text_to)
            if line_position == -1:
                sys.exit(f'the_span_text not found in the line {the_span_text_to},{the_line_num_to}')
            global_start_to = curr_line_beginning_pos + line_position
            global_end_to = global_start_to + len(the_span_text_to)
            assert the_literature[global_start_to:global_end_to] == the_span_text_to
        curr_line_beginning_pos = curr_line_beginning_pos + len(aline) + 1
    return global_start_from, global_end_from, global_start_to, global_end_to


def chapterize_and_get_textpos_in_chapter(the_literature, global_start_from, global_end_from, textspan_from, global_start_to, global_end_to, textspan_to):
    # obtain the chapterized literature
    string_number_uppercase = '(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|TWENTY-ONE|TWENTY-TWO|TWENTY-THREE|TWENTY-FOUR|TWENTY-FIVE|TWENTY-SIX|TWENTY-SEVEN|TWENTY-EIGHT|TWENTY-NINE|THIRTY|THIRTY-ONE|THIRTY-TWO|THIRTY-THREE|THIRTY-FOUR|THIRTY-FIVE|THIRTY-SIX|THIRTY-SEVEN|THIRTY-EIGHT|THIRTY-NINE|FORTY|FORTY-ONE|FORTY-TWO|FORTY-THREE|FORTY-FOUR|FORTY-FIVE|FORTY-SIX|FORTY-SEVEN|FORTY-EIGHT|FORTY-NINE|FIFTY)'
    string_number_lowercase = '(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six|twenty-seven|twenty-eight|twenty-nine|thirty|thirty-one|thirty-two|thirty-three|thirty-four|thirty-five|thirty-six|thirty-seven|thirty-eight|thirty-nine|forty|forty-one|forty-two|forty-three|forty-four|forty-five|forty-six|forty-seven|forty-eight|forty-nine|fifty)'
    string_number_FirstChar_uppercase = '(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Twenty-One|Twenty-Two|Twenty-Three|Twenty-Four|Twenty-Five|Twenty-Six|Twenty-Seven|Twenty-Eight|Twenty-Nine|Thirty|Thirty-One|Thirty-Two|Thirty-Three|Thirty-Four|Thirty-Five|Thirty-Six|Thirty-Seven|Thirty-Eight|Thirty-Nine|Forty|Forty-One|Forty-Two|Forty-Three|Forty-Four|Forty-Five|Forty-Six|Forty-Seven|Forty-Eight|Forty-Nine|Fifty)'
    string_number_ordered_form_with_the = '(THE FIRST|THE SECOND|THE THIRD|THE FOURTH|THE FIFTH|THE SIXTH|THE SEVENTH|THE EIGHTH|THE NINTH|THE TENTH|THE ELEVENTH|THE TWELFTH|THE THIRTEENTH|THE FOURTEENTH|THE FIFTEENTH|THE SIXTEENTH|THE SEVENTEENTH|THE EIGHTEENTH|THE NINETEENTH|THE TWENTIETH|THE TWENTY-FIRST|THE TWENTY-SECOND|THE TWENTY-THIRD|THE TWENTY-FOURTH|THE TWENTY-FIFTH|THE TWENTY-SIXTH|THE TWENTY-SEVENTH|THE TWENTY-EIGHTH|THE TWENTY-NINTH|THE THIRTIETH|THE THIRTY-FIRST|THE THIRTY-SECOND|THE THIRTY-THIRD|THE THIRTY-FOURTH|THE THIRTY-FIFTH|THE THIRTY-SIXTH|THE THIRTY-SEVENTH|THE THIRTY-EIGHTH|THE THIRTY-NINTH|THE FORTIETH|THE FORTY-FIRST|THE FORTY-SECOND|THE FORTY-THIRD|THE FORTY-FOURTH|THE FORTY-FIFTH|THE FORTY-SIXTH|THE FORTY-SEVENTH|THE FORTY-EIGHTH|THE FORTY-NINTH|THE FIFTIETH)'
    string_number_ordered_form = '(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|SEVENTEENTH|EIGHTEENTH|NINETEENTH|TWENTIETH|TWENTY-FIRST|TWENTY-SECOND|TWENTY-THIRD|TWENTY-FOURTH|TWENTY-FIFTH|TWENTY-SIXTH|TWENTY-SEVENTH|TWENTY-EIGHTH|TWENTY-NINTH|THIRTIETH|THIRTY-FIRST|THIRTY-SECOND|THIRTY-THIRD|THIRTY-FOURTH|THIRTY-FIFTH|THIRTY-SIXTH|THIRTY-SEVENTH|THIRTY-EIGHTH|THIRTY-NINTH|FORTIETH|FORTY-FIRST|FORTY-SECOND|FORTY-THIRD|FORTY-FOURTH|FORTY-FIFTH|FORTY-SIXTH|FORTY-SEVENTH|FORTY-EIGHTH|FORTY-NINTH|FIFTIETH)'
    string_number_ordered_form_FirstChar_uppercase = '(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth|Sixteenth|Seventeenth|Eighteenth|Nineteenth|Twentieth|Twenty-First|Twenty-Second|Twenty-Third|Twenty-Fourth|Twenty-Fifth|Twenty-Sixth|Twenty-Seventh|Twenty-Eighth|Twenty-Ninth|Thirtieth|Thirty-First|Thirty-Second|Thirty-Third|Thirty-Fourth|Thirty-Fifth|Thirty-Sixth|Thirty-Seventh|Thirty-Eighth|Thirty-Ninth|Fortieth|Forty-First|Forty-Second|Forty-Third|Forty-Fourth|Forty-Fifth|Forty-Sixth|Forty-Seventh|Forty-Eighth|Forty-Ninth|Fiftieth)'
    # re_count_chapter_pattern_1 = re.compile('^CHAPTER ONE$', re.MULTILINE)
    re_count_chapter_pattern_1 = re.compile(('^\s*(chapter|Chapter|CHAPTER)\s*'
                                           '(((?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
                                           '|((?=[MDCLXVI])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
                                           '|([0-9]+)'
                                           '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + '|' + string_number_ordered_form_with_the + '|' + string_number_ordered_form + '|' + string_number_ordered_form_FirstChar_uppercase + ')'
                                           '\s*(:|\.|~| |\w)*\s*$'), re.MULTILINE)
    # re_count_chapter_pattern_1 = re.compile(('^\s*(chapter|Chapter|CHAPTER)\s*'
    #                                     '(((?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
    #                                     '|((?=[MDCLXVI])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
    #                                     '|([0-9]+)'
    #                                     '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + ')'
    #                                     '\s*(:|\.|~)*\s*$'), re.MULTILINE)
    re_count_chapter_pattern_2 = re.compile(('^\s*(scene|Scene|SCENE)\s*'
                                           '(((?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
                                           '|((?=[MDCLXVI])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
                                           '|([0-9]+)'
                                           '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + '|' + string_number_ordered_form_with_the + '|' + string_number_ordered_form + '|' + string_number_ordered_form_FirstChar_uppercase + ')'
                                           '\s*(:|\.)*\s*$'), re.MULTILINE)
    re_count_chapter_pattern_3 = re.compile(('^(part|Part|PART|Book|BOOK|ACT)\s*'
                                           '(((?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
                                           '|((?=[MDCLXVI])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
                                           '|([0-9]+)'
                                           '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + '|' + string_number_ordered_form_with_the + '|' + string_number_ordered_form + '|' + string_number_ordered_form_FirstChar_uppercase + ')'
                                           '\s*(:|\.)*\s*$'), re.MULTILINE)
    re_count_chapter_pattern_4 = re.compile(('^('
                                             '(</b>)?(<b>)?\s*'
                                             '((?=[MDCLXVI])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
                                             '|(?=[MDCLXVI])(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
                                             '|([0-9]+)'
                                             '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + '|' + string_number_ordered_form_with_the + '|' + string_number_ordered_form + '|' + string_number_ordered_form_FirstChar_uppercase + ')'
                                             '\.\s*(\w|“|”|‘|’|\'|\"|-|\.| |,|!|\?|:|;|~)*\.?'
                                             ')$'), re.MULTILINE)
                                            #  '\.\s*(\w|“|”|‘|’|\'|\"|-|\.)*\.?'
                                            # '\..*'
    # re_count_chapter_pattern_4 = re.compile(('^('
    #                                          '('
    #                                          '<b>\s*'
    #                                          '((?=[MDCLXVI])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))'
    #                                          '|(?=[MDCLXVI])(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))'
    #                                          '|([0-9]+)'
    #                                          '|' + string_number_uppercase + '|' + string_number_lowercase + '|' + string_number_FirstChar_uppercase + ')'
    #                                          '\.\s*(\w|“|”|‘|’|\'|\"|-|\.| |,|!|\?|:|;)*\.?'
    #                                          ')'
    #                                          '|([0-9]+'
    #                                          ')'
    #                                          ')$'), re.MULTILINE)
    #                                         #  '\.\s*(\w|“|”|‘|’|\'|\"|-|\.)*\.?'
    #                                         # '\..*'
    
    gutenberg_string_start_pos = the_literature.find('1.B. “Project Gutenberg” is a registered trademark. It may only be')
    if gutenberg_string_start_pos != -1:
        the_literature = the_literature[:gutenberg_string_start_pos]
    gutenberg_string_start_pos = the_literature.find('1.A. By reading or using any part of this Project Gutenberg-tm')
    if gutenberg_string_start_pos != -1:
        the_literature = the_literature[:gutenberg_string_start_pos]

    lst_chapters = list()
    
    findall_iter = re_count_chapter_pattern_1.finditer(the_literature)
    chapter_index = 0
    zero_chapter_match = next(findall_iter, None)
    the_textspan_pos_in_chapter_from = None
    the_textspan_pos_in_chapter_to = None
    the_textspan_in_chapter_index_from = None
    the_textspan_in_chapter_index_to = None

    if zero_chapter_match is not None:
        this_chapter = the_literature[:zero_chapter_match.start(0)]
        this_chapter_idx = zero_chapter_match.start(0)
        # print(f'Chapter from chapter_index = {chapter_index}')
        # print(the_zero_chapter)
        if global_start_from >= 0 and global_end_from <= this_chapter_idx:
            the_textspan_pos_in_chapter_from = (global_start_from - 0, global_end_from - 0)
            the_textspan_in_chapter_index_from = 0
            assert the_literature[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
        if global_start_to >= 0 and global_end_to <= this_chapter_idx:
            the_textspan_pos_in_chapter_to = (global_start_to - 0, global_end_to - 0)
            the_textspan_in_chapter_index_to = 0
            assert the_literature[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
        lst_chapters.append((0, this_chapter))
        the_last_chapter_idx = this_chapter_idx
        for aresult in findall_iter:
            this_chapter_idx = aresult.start(0)
            this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
            if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                the_textspan_in_chapter_index_from = chapter_index + 1
                assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
            if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                the_textspan_in_chapter_index_to = chapter_index + 1
                assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
            the_last_chapter_idx = this_chapter_idx
            chapter_index += 1
            lst_chapters.append((chapter_index, this_chapter))

        this_chapter = the_literature[the_last_chapter_idx:]
        chapter_index += 1
        lst_chapters.append((chapter_index, this_chapter))
        if global_start_from >= the_last_chapter_idx and global_end_from <= len(the_literature):
            the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
            the_textspan_in_chapter_index_from = chapter_index
            assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
        if global_start_to >= the_last_chapter_idx and global_end_to <= len(the_literature):
            the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
            the_textspan_in_chapter_index_to = chapter_index
            assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
        # print(global_start_to, global_end_to)
        # print(the_last_chapter_idx, this_chapter_idx)
        # assert the_textspan_pos_in_chapter_from is not None
        # assert the_textspan_pos_in_chapter_to is not None
    else:
        findall_iter = re_count_chapter_pattern_2.finditer(the_literature)
        zero_chapter_match = next(findall_iter, None)
        if zero_chapter_match is not None:
            this_chapter = the_literature[:zero_chapter_match.start(0)]
            this_chapter_idx = zero_chapter_match.start(0)
            if global_start_from >= 0 and global_end_from <= this_chapter_idx:
                the_textspan_pos_in_chapter_from = (global_start_from - 0, global_end_from - 0)
                the_textspan_in_chapter_index_from = 0
                assert the_literature[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
            if global_start_to >= 0 and global_end_to <= this_chapter_idx:
                the_textspan_pos_in_chapter_to = (global_start_to - 0, global_end_to - 0)
                the_textspan_in_chapter_index_to = 0
                assert the_literature[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
            lst_chapters.append((0, this_chapter))
            the_last_chapter_idx = this_chapter_idx
            for aresult in findall_iter:
                this_chapter_idx = aresult.start(0)
                this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
                if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                    the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                    the_textspan_in_chapter_index_from = chapter_index + 1
                    assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                    the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                    the_textspan_in_chapter_index_to = chapter_index + 1
                    assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                the_last_chapter_idx = this_chapter_idx
                chapter_index += 1
                lst_chapters.append((chapter_index, this_chapter))
            # the last chapter until the end
            this_chapter = the_literature[the_last_chapter_idx:]
            chapter_index += 1
            lst_chapters.append((chapter_index, this_chapter))
            if global_start_from >= the_last_chapter_idx and global_end_from <= len(the_literature):
                the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                the_textspan_in_chapter_index_from = chapter_index
                assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
            if global_start_to >= the_last_chapter_idx and global_end_to <= len(the_literature):
                the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                the_textspan_in_chapter_index_to = chapter_index
                assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
        else:
            findall_iter = re_count_chapter_pattern_3.finditer(the_literature)
            zero_chapter_match = next(findall_iter, None)
            if zero_chapter_match is not None:
                this_chapter = the_literature[:zero_chapter_match.start(0)]
                this_chapter_idx = zero_chapter_match.start(0)
                if global_start_from >= 0 and global_end_from <= this_chapter_idx:
                    the_textspan_pos_in_chapter_from = (global_start_from - 0, global_end_from - 0)
                    the_textspan_in_chapter_index_from = 0
                    assert the_literature[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                if global_start_to >= 0 and global_end_to <= this_chapter_idx:
                    the_textspan_pos_in_chapter_to = (global_start_to - 0, global_end_to - 0)
                    the_textspan_in_chapter_index_to = 0
                    assert the_literature[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                lst_chapters.append((0, this_chapter))
                the_last_chapter_idx = this_chapter_idx
                for aresult in findall_iter:
                    this_chapter_idx = aresult.start(0)
                    this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
                    if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                        the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                        the_textspan_in_chapter_index_from = chapter_index + 1
                        assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                    if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                        the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                        the_textspan_in_chapter_index_to = chapter_index + 1
                        assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                    the_last_chapter_idx = this_chapter_idx
                    chapter_index += 1
                    lst_chapters.append((chapter_index, this_chapter))
                # the last chapter until the end
                this_chapter = the_literature[the_last_chapter_idx:]
                chapter_index += 1
                lst_chapters.append((chapter_index, this_chapter))
                if global_start_from >= the_last_chapter_idx and global_end_from <= len(the_literature):
                    the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                    the_textspan_in_chapter_index_from = chapter_index
                    assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                if global_start_to >= the_last_chapter_idx and global_end_to <= len(the_literature):
                    the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                    the_textspan_in_chapter_index_to = chapter_index
                    assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
            else:
                findall_iter = re_count_chapter_pattern_4.finditer(the_literature)
                zero_chapter_match = next(findall_iter, None)
                if zero_chapter_match is not None:
                    this_chapter = the_literature[:zero_chapter_match.start(0)]
                    this_chapter_idx = zero_chapter_match.start(0)
                    if global_start_from >= 0 and global_end_from <= this_chapter_idx:
                        the_textspan_pos_in_chapter_from = (global_start_from - 0, global_end_from - 0)
                        the_textspan_in_chapter_index_from = 0
                        assert the_literature[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                    if global_start_to >= 0 and global_end_to <= this_chapter_idx:
                        the_textspan_pos_in_chapter_to = (global_start_to - 0, global_end_to - 0)
                        the_textspan_in_chapter_index_to = 0
                        assert the_literature[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                    lst_chapters.append((0, this_chapter))
                    the_last_chapter_idx = this_chapter_idx
                    for aresult in findall_iter:
                        this_chapter_idx = aresult.start(0)
                        this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
                        if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                            the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                            the_textspan_in_chapter_index_from = chapter_index + 1
                            assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                        if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                            the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                            the_textspan_in_chapter_index_to = chapter_index + 1
                            assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                        the_last_chapter_idx = this_chapter_idx
                        chapter_index += 1
                        lst_chapters.append((chapter_index, this_chapter))
                    # the last chapter until the end
                    this_chapter = the_literature[the_last_chapter_idx:]
                    chapter_index += 1
                    lst_chapters.append((chapter_index, this_chapter))
                    if global_start_from >= the_last_chapter_idx and global_end_from <= len(the_literature):
                        the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                        the_textspan_in_chapter_index_from = chapter_index
                        assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                    if global_start_to >= the_last_chapter_idx and global_end_to <= len(the_literature):
                        the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                        the_textspan_in_chapter_index_to = chapter_index
                        assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                else:
                    the_literature_tokens = the_literature.split(' ')
                    len_the_literature_tokens = len(the_literature_tokens)
                    
                    curr_token_num = 0
                    this_chapter_idx = 0
                    the_last_chapter_idx = 0
                    while curr_token_num < len_the_literature_tokens:
                        if curr_token_num + 8192 < len_the_literature_tokens:
                            this_chapter_tokens = the_literature_tokens[curr_token_num:curr_token_num+8192]
                            temp_chapter_idx = 0
                            for atoken in this_chapter_tokens:
                                temp_chapter_idx += len(atoken) + 1
                            this_chapter_idx += temp_chapter_idx
                            this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
                            lst_chapters.append((chapter_index, this_chapter))
                            chapter_index += 1
                            assert the_literature[the_last_chapter_idx:this_chapter_idx] == ' '.join(this_chapter_tokens) + ' '
                            if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                                the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                                the_textspan_in_chapter_index_from = chapter_index - 1
                                assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                            if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                                the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                                the_textspan_in_chapter_index_to = chapter_index - 1
                                assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                            the_last_chapter_idx = this_chapter_idx
                            curr_token_num += 8192
                        else:
                            this_chapter_tokens = the_literature_tokens[curr_token_num:]
                            temp_chapter_idx = 0
                            for atoken in this_chapter_tokens:
                                temp_chapter_idx += len(atoken) + 1
                            temp_chapter_idx -= 1
                            this_chapter_idx += temp_chapter_idx
                            this_chapter = the_literature[the_last_chapter_idx:this_chapter_idx]
                            lst_chapters.append((chapter_index, this_chapter))
                            assert the_literature[the_last_chapter_idx:this_chapter_idx] == ' '.join(this_chapter_tokens)
                            assert this_chapter_idx == len(the_literature)
                            if global_start_from >= the_last_chapter_idx and global_end_from <= this_chapter_idx:
                                the_textspan_pos_in_chapter_from = (global_start_from - the_last_chapter_idx, global_end_from - the_last_chapter_idx)
                                the_textspan_in_chapter_index_from = chapter_index
                                assert this_chapter[the_textspan_pos_in_chapter_from[0]:the_textspan_pos_in_chapter_from[1]] == textspan_from
                            if global_start_to >= the_last_chapter_idx and global_end_to <= this_chapter_idx:
                                the_textspan_pos_in_chapter_to = (global_start_to - the_last_chapter_idx, global_end_to - the_last_chapter_idx)
                                the_textspan_in_chapter_index_to = chapter_index
                                assert this_chapter[the_textspan_pos_in_chapter_to[0]:the_textspan_pos_in_chapter_to[1]] == textspan_to
                            the_last_chapter_idx = this_chapter_idx
                            assert curr_token_num + len(this_chapter_tokens) == len_the_literature_tokens
                            break
    # print(textspan_from, textspan_to)
    assert the_textspan_pos_in_chapter_from is not None
    assert the_textspan_pos_in_chapter_to is not None
    assert the_textspan_in_chapter_index_from is not None
    assert the_textspan_in_chapter_index_to is not None

    return lst_chapters, the_textspan_pos_in_chapter_from, the_textspan_pos_in_chapter_to, the_textspan_in_chapter_index_from, the_textspan_in_chapter_index_to


if __name__ == '__main__':
    # back to the scene.
    dct_fname2literature = read_literatures('./narrativeqa/tmp/')
    fname_annotations = './narrativeqaIntegrated.csv'
    fpIn_annotations = open(fname_annotations, 'rt', encoding='utf8')
    csvreader = csv.reader(fpIn_annotations, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    row0 = next(csvreader)

    fpOut_restructured_annotations = open('./narrativeqaRestructuredAnnotations.csv', 'wt', encoding='utf8')
    csvwriter_restructured_annotations = csv.writer(fpOut_restructured_annotations, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csvwriter_restructured_annotations.writerow(['document_id','set','question','answer1','answer2','question_tokenized','answer1_tokenized','answer2_tokenized','question_span_grounded','question_linenum','answer_span_grounded','answer_linenum', 'mention1', 'mention1_globalpos', 'mention1_chapterpos', 'mention1_chapterindex', 'mention2', 'mention2_globalpos', 'mention2_chapterpos', 'mention2_chapterindex'])
    fpOut_chapters = open('./narrativeqaChapters.csv', 'wt', encoding='utf8')
    csvwriter_chapters = csv.writer(fpOut_chapters, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csvwriter_chapters.writerow(['document_id','chapter_index','chapter_text'])

    for row_i, arow in enumerate(csvreader):
        # '105672e2bb1bcc9dd50e90822ca49a66876ac198'
        # '145200abf14baeffa646797dfbfa58861cb4b079'
        # 16380f54b88ffa64ca4cd82d3dc5481e78ad5118
        # 15329a294f5c4e40208bc75102fbd312f58599cb
        # 193805049ab9d8f8f745cee9d77d8d2ea8cc9a2d
        # 229bcf6a72dbf68b2b074ab4266fa677da43f68d
        the_literature = dct_fname2literature[arow[0]]

        global_start_from, global_end_from, global_start_to, global_end_to = find_global_pos(the_literature, arow[8], int(arow[9]), arow[10], int(arow[11]))
        lst_chapters, the_textspan_pos_in_chapter_from, the_textspan_pos_in_chapter_to, the_textspan_in_chapter_index_from, the_textspan_in_chapter_index_to = chapterize_and_get_textpos_in_chapter(the_literature, global_start_from, global_end_from, arow[8], global_start_to, global_end_to, arow[10])
        csvwriter_restructured_annotations.writerow(arow + [arow[8], f'{global_start_from}:{global_end_from}', f'{the_textspan_pos_in_chapter_from[0]}:{the_textspan_pos_in_chapter_from[1]}', str(the_textspan_in_chapter_index_from), arow[10], f'{global_start_to}:{global_end_to}', f'{the_textspan_pos_in_chapter_to[0]}:{the_textspan_pos_in_chapter_to[1]}', str(the_textspan_in_chapter_index_to)])
        for chapter_i, chapter_text in lst_chapters:
            csvwriter_chapters.writerow([arow[0], chapter_i, chapter_text])
    
    fpOut_restructured_annotations.close()
    fpOut_chapters.close()
    fpIn_annotations.close()