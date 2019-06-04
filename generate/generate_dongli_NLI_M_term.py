import os
from nltk.tokenize import word_tokenize
import pandas as pd

data_dir = '../data/dongli/'
test_file = "dongli_test.csv"

train_file = "dongli_train.csv"

dir_path = data_dir + 'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

sent_id = 0
with open(dir_path + "test_term_NLI_M.csv", "w", encoding="utf-8") as g:
    test = pd.read_csv(data_dir + test_file, index_col=0)
    test = [tuple(x) for x in test.values]
    for text, term, polarity in test:
        line = '{}\t{}\t{}\t{}\n'.format(sent_id, polarity, term, text)
        g.write(line)
        for token in word_tokenize(text):
            if token not in term:
                line = '{}\t{}\t{}\t{}\n'.format(sent_id, 'none', term, text)
                g.write(line)
        sent_id += 1

with open(dir_path + "train_term_NLI_M.csv", "w", encoding="utf-8") as g:
    train = pd.read_csv(data_dir + train_file, index_col=0)
    train = [tuple(x) for x in train.values]
    for text, term, polarity in train:
        line = '{}\t{}\t{}\t{}\n'.format(sent_id, polarity, term, text)
        g.write(line)
        for token in word_tokenize(text):
            if token not in term:
                line = '{}\t{}\t{}\t{}\n'.format(sent_id, 'none', term, text)
                g.write(line)
        sent_id += 1

#     with open(data_dir + test_file, "r", encoding="utf-8") as f:
#         s = f.readline().strip()
#         while s:
#             term = []
#             polarity = []
#             if "<sentence id" in s:
#                 left = s.find("id")
#                 right = s.find(">")
#                 id = s[left + 4:right - 1]
#                 while not "</sentence>" in s:
#                     if "<text>" in s:
#                         left = s.find("<text>")
#                         right = s.find("</text>")
#                         text = s[left + 6:right]
#                     if "aspectTerm" in s and "aspectTerms" not in s:
#                         left = s.find("term=")
#                         right = s.find("polarity=")
#                         term.append(s[left + 6:right - 2])
#                         left = s.find("polarity=")
#                         right = s.find("from=")
#                         polarity.append(s[left + 10:right - 3])
#                     s = f.readline().strip()
#                 for te in term:
#                     g.write(id + "\t" + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")
#
#                 for token in word_tokenize(text):
#                     if token not in term:
#                         g.write(id + "\t" + "none" + "\t" + token + "\t" + text + "\n")
#
#             else:
#                 s = f.readline().strip()
#
# with open(dir_path + "train_term_NLI_M.csv", "w", encoding="utf-8") as g:
#     with open(data_dir + train_file, "r", encoding="utf-8") as f:
#         s = f.readline().strip()
#         while s:
#             term = []
#             polarity = []
#             if "<sentence id" in s:
#                 left = s.find("id")
#                 right = s.find(">")
#                 id = s[left + 4:right - 1]
#                 while not "</sentence>" in s:
#                     if "<text>" in s:
#                         left = s.find("<text>")
#                         right = s.find("</text>")
#                         text = s[left + 6:right]
#                     if "aspectTerm" in s and "aspectTerms" not in s:
#                         left = s.find("term=")
#                         right = s.find("polarity=")
#                         term.append(s[left + 6:right - 2])
#                         left = s.find("polarity=")
#                         right = s.find("from=")
#                         polarity.append(s[left + 10:right - 2])
#
#                     s = f.readline().strip()
#                 for te in term:
#                     g.write(id + "\t" + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")
#
#                 for token in word_tokenize(text):
#                     if token not in term:
#                         g.write(id + "\t" + "none" + "\t" + token + "\t" + text + "\n")
#
#             else:
#                 s = f.readline().strip()
