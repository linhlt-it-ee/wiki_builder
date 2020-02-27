'''@HOW TO USE: extract_name_mentions_from_claims '''
'''@excel_file: excel file of patent claims '''
'''@max_n_gram_range: should only be maximum=4 because there are often maximum 4 words in wiki entity'''
'''@name_mention_folder: where list of name mentions is splitted into smaller file for easier crawling'''
'''@@@WARNING there is not '/' at the end of name_mention_folder'''
import os
import utils.text_util as text_util
import utils.file_util as file_util
import sys
def extract_name_mentions_from_claims(excel_file, sheet_name, max_n_gram_range, name_mention_folder,number_of_files):

    claims = file_util.get_claims(excel_file, sheet_name)
    name_mentions = text_util.get_name_mention_from_claims_nltk(claims)
    n_gram_name_mentions = text_util.generate_n_gram_from_name_mentions(name_mentions, max_n_gram_range)
    split_name_mention_list(n_gram_name_mentions, name_mention_folder,number_of_files)


'''@HOW TO USE: search_wiki_with_threads_v3 function'''
'''@name_mention_list:a list of name mentions from claims '''


def split_name_mention_list(name_mention_list, name_mention_folder,number_of_files):
    total = len(name_mention_list)
    line_num_each_file = int(total / number_of_files)
    file_util.mkdir(name_mention_folder)
    for i in range(0, number_of_files):
        new_file = open(name_mention_folder + "/name_mention_" + str(i) + ".txt", "w")
        if i * line_num_each_file + line_num_each_file < total - 1:
            start = i * line_num_each_file
            end = i * line_num_each_file + line_num_each_file
            print(start, end)
            for j in name_mention_list[start:end]:
                new_file.write(j+'\n')
        else:
            for j in name_mention_list[i * line_num_each_file:]:
                new_file.write(j+'\n')
        new_file.close()

if __name__ == "__main__":
    #python3 sony_patent_evaluation/test/extract_name_mentions.py "patent_claims.xlsx"│ 1 file changed, 2 insertions(+), 2 deletions(-) "Sheet1" 4 "entity_folder_03122019" 20

    extract_name_mentions_from_claims(sys.argv[1:][0], sys.argv[1:][1], int(sys.argv[1:][2]), sys.argv[1:][3],int(sys.argv[1:][4]))