import os
from nltk.tokenize import word_tokenize
import pandas as pd

data_dir = '../data/indo_peter/'
test_file = "test.xml"

train_file = "train.xml"

dir_path = data_dir + 'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(dir_path + "test_term_NLI_M.csv", "w", encoding="utf-8") as g:
    with open(data_dir + test_file, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            term = []
            polarity = []
            if "<sentence id" in s:
                left = s.find("id")
                right = s.find(">")
                id = 'test_'+s[left + 4:right - 1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left = s.find("<text>")
                        right = s.find("</text>")
                        text = s[left + 6:right]
                    if "aspectTerm" in s and "aspectTerms" not in s:
                        left = s.find("polarity=")
                        right = s.find("term=")
                        polarity.append(s[left + 10:right - 2])
                        left = s.find("term=")
                        right = s.find("to=")
                        term.append(s[left+6:right-2])
                    s = f.readline().strip()
                for te in term:
                    g.write(id + "\t" + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")

                for token in word_tokenize(text):
                    if token not in term:
                        g.write(id + "\t" + "none" + "\t" + token + "\t" + text + "\n")

            else:
                s = f.readline().strip()

with open(dir_path + "train_term_NLI_M.csv", "w", encoding="utf-8") as g:
    with open(data_dir + train_file, "r", encoding="utf-8") as f:
        s = f.readline().strip()
        while s:
            term = []
            polarity = []
            if "<sentence id" in s:
                left = s.find("id")
                right = s.find(">")
                id = 'train_'+s[left + 4:right - 1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left = s.find("<text>")
                        right = s.find("</text>")
                        text = s[left + 6:right]
                    if "aspectTerm" in s and "aspectTerms" not in s:
                        left = s.find("polarity=")
                        right = s.find("term=")
                        polarity.append(s[left + 10:right - 2])
                        left = s.find("term=")
                        right = s.find("to=")
                        term.append(s[left + 6:right - 2])

                    s = f.readline().strip()
                for te in term:
                    g.write(id + "\t" + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")

                for token in word_tokenize(text):
                    if token not in term:
                        g.write(id + "\t" + "none" + "\t" + token + "\t" + text + "\n")

            else:
                s = f.readline().strip()
