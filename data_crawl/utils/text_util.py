import os
import operator
import re
import nltk
import copy
import utils.sony_common as sony_const
from nltk.corpus import words as nltk_words
from pattern.text.en import singularize
from janome.tokenfilter import CompoundNounFilter, POSKeepFilter, POSStopFilter
from janome.analyzer import Analyzer
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
token_filters = [POSKeepFilter(['名詞']), POSStopFilter(['記号','助詞', '名詞,数'])]
analyzer1 = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
token_filters = [CompoundNounFilter(), POSStopFilter(['記号','助詞', '名詞,数'])]
analyzer2 = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
    
    
def convert_plural_to_singular(text):
    return singularize(text)

def get_range_words(text):
    return re.findall("(?:minimum|maximum|offset|ratio|axis|width|length)\s+",text)

def get_negative_words(text):
    words = get_pattern_from_text_nltk("JJ",text)
    negative_words=[]
    for word in words:
        positions, match_groups= match_with_position("^un|^non|^de|^dis|^anti|^im|^il|^ir|^a",word)
        if len(positions)>0:
            begin_indices=[i for i,x in enumerate(positions) if x==0]
            match_group=match_groups[begin_indices[0]]
            root_index=len(match_group)
            if root_index>0:
                root_word=word[root_index:]
                if len(root_word)>1:
                    if root_word in nltk_words.words():
                        negative_words.append(word)
    return negative_words

def remove_characters(removed_regex,regex_converted_to, input_text):
    return re.sub(removed_regex,regex_converted_to,input_text)

def nltk_tokenize(text):
    pos_tag_list = nltk.pos_tag(text.split())
    words = [x[0] for x in pos_tag_list]
    tags = [x[1] for x in pos_tag_list]
    return words, tags

def match_with_position(pattern, sentence):
    p = re.compile(pattern)
    positions = []
    matched_groups = []
    for m in p.finditer(sentence):
        # print(m.start(), m.group())
        positions.append(m.start())
        # print(sentence[0:m.start()-1])
        matched_groups.append(m.group())
    return positions, matched_groups

def get_position_from_pos_tag(tag_pos, tag_phrases, words, tag_str):
    token_count = len(tag_phrases.split('#'))
    token_count = token_count - 1 if tag_phrases.endswith("#") else token_count
    consider_text = tag_str[0:tag_pos + 1]
    tokens_before = len(consider_text.split('#'))
    tokens_before = 0 if tokens_before == 1 or (tokens_before == 2 and consider_text.endswith('#')) else tokens_before
    tokens_before = tokens_before - 1 if tag_str[0:tag_pos].endswith("#") else tokens_before
    related_words = words[tokens_before:tokens_before + token_count]
    related_words = [re.sub('<.*>|”|“', '', x) for x in related_words]
    related_words = [re.sub('-', ' ', x) for x in related_words]
    name_mention = " ".join(related_words)
    return name_mention

def not_special_chars_inside(text):
    return len(re.findall("^[_A-z]*((-|\s)*[_A-z])*$", text)) > 0

def get_pattern_from_text_nltk(pattern, text):
    words, tags = nltk_tokenize(str(text))
    words = [re.sub('<.*>|”|“', '', x) for x in words]
    pos_punct = ["VBD" if x in sony_const.noun_tag_list and words[i] in sony_const.include_verbs else x for i, x in enumerate(tags)]
    pos_punct = ["CC" if (x in sony_const.noun_tag_list or x in sony_const.other_tag_list) and words[i] in sony_const.conjunctions else x for i, x in
                 enumerate(pos_punct)]
    pos_punct = ["UNK" if (x in sony_const.other_tag_list or x in sony_const.noun_tag_list) and words[i] in sony_const.no_meaning_words else x for i, x
                 in enumerate(pos_punct)]
    pos_punct = ["UNK" if not not_special_chars_inside(words[i]) or len(words[i]) == 1 else x for i, x in
                 enumerate(pos_punct)]
    post_punct_str = "#".join(pos_punct)
    tag_pos, matched_tag_groups = match_with_position(pattern, post_punct_str)
    matched_texts = []
    str_to_match = post_punct_str
    
    for j, matched in enumerate(matched_tag_groups):
        matched_text = get_position_from_pos_tag(tag_pos[j], matched, words, str_to_match)
        other_words = re.findall('\(.*\)', matched_text)
        other_words = [re.sub(sony_const.brackets, '', x) for x in other_words]
        if len(other_words) > 0:
            matched_text = re.sub("|".join(other_words), '', matched_text)
            matched_texts.extend(other_words)
        matched_text = re.sub(sony_const.brackets, '', matched_text)
        number_words = ['^' + x + '\s+' for x in sony_const.number_words]
        matched_text = re.sub('|'.join(number_words), '', matched_text)
        matched_text = re.sub('^(?:\w\s+)+|(?:\s+\w)+$', '', matched_text)
        matched_text = re.sub('\s{2,}', ' ', matched_text)
        matched_texts.append(matched_text)
    matched_texts = [x.strip() for x in matched_texts]
    
    return matched_texts

def get_pattern_from_text_ja(pattern, text):
    global analyzer1, analyzer2
    matched_texts = [token.surface for token in analyzer1.analyze(text)]
    matched_texts.extend([token.surface for token in analyzer2.analyze(text)])

    return matched_texts

def get_name_mention_from_claims_nltk(claims, lang="en"):
    name_mention_pattern = "(?:JJ#|NN#|NNS#|NNP#|NNPS#)*(?:NN|NNS|NNP|NNPS)+"
    name_mentions = []
    for i, claim in enumerate(claims):
        short_texts = re.split(',|;|:|\.', str(claim))
        for short_text in short_texts:
            if lang == "en":
                targets = get_pattern_from_text_nltk(name_mention_pattern, short_text)
            elif lang == "ja":
                targets = get_pattern_from_text_ja(name_mention_pattern, short_text)
            else:
                raise NotImplementedError
                
            name_mentions.extend(targets)
    
    if lang == "en":
        name_mentions = [x for x in name_mentions if len(x.split()) >= 2 or (len(x.split()) == 1 and not_special_chars_inside(x) and len(x) > 1)]
    name_mentions = list(set(name_mentions))
    
    return name_mentions


def generate_ngrams(s, n, lang="en"):
    # Convert to lowercases
    if lang == "en":
        s = s.lower()
        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
    elif lang == "ja":
        tokens = s
    else:
        raise NotImplementedError
           
    ngrams = []
    for x in zip(*[tokens[i:] for i in range(n)]):
        ngram = " ".join(x) if lang == "en" else "".join(x)
        try:
            float(ngram)
        except:
            ngrams.append(ngram)
    # ngrams = [" ".join(ngram) for ngram in ngrams_generator]
    return ngrams

def generate_n_gram_from_name_mentions(name_mentions,n_gram_range, lang="en"):
    all_name_mention=[]
    for name_mention in name_mentions:
        for i in range(n_gram_range):
            ngram_count=i+1
            ngram_list=generate_ngrams(name_mention,ngram_count, lang)
            all_name_mention.extend(ngram_list)
            all_name_mention.append(name_mention)
    all_name_mention=list(set(all_name_mention))
    return all_name_mention

text="nonplanar defect amorphous an antifuse varactor formed on the substrate structure, the antifuse varactor having a third gate terminal"
print(get_negative_words(text))
print(get_range_words(text))