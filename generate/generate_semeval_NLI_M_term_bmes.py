import os
from nltk.tokenize import word_tokenize

for domain in ['Restaurants', 'laptop']:
    data_dir = '../data/semeval2014/'
    test_file = "Restaurants_Test_Gold.xml" if domain == 'Restaurants' else 'Laptops_Test_Gold.xml'

    train_file = "Restaurants_Train.xml" if domain == 'Restaurants' else 'Laptop_Train_v2.xml'

    dir_path = data_dir + 'bert-pair-bmes/{}/'.format(domain)
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
                    id = s[left + 4:right - 1]
                    while not "</sentence>" in s:
                        if "<text>" in s:
                            left = s.find("<text>")
                            right = s.find("</text>")
                            text = s[left + 6:right]
                        if "aspectTerm" in s and "aspectTerms" not in s:
                            left = s.find("term=")
                            right = s.find("polarity=")
                            term.append(s[left + 6:right - 2])
                            left = s.find("polarity=")
                            right = s.find("from=")
                            polarity.append(s[left + 10:right - 3])

                            # polarity.append(s[left + 10:right - 3])

                        s = f.readline().strip()

                    for te in term:
                        te_token = te.split(' ')
                        if len(te_token) == 1:
                            g.write(id + "\t" + 'single ' + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")
                        else:
                            for i, t in enumerate(te_token):
                                if i == 0:
                                    g.write(id + "\t" + 'first word ' + polarity[
                                        term.index(te)] + "\t" + t + "\t" + text + "\n")
                                elif i == len(te_token) - 1:
                                    g.write(
                                        id + "\t" + 'last word ' + polarity[
                                            term.index(te)] + "\t" + t + "\t" + text + "\n")
                                else:
                                    g.write(id + "\t" + 'middle word ' + polarity[
                                        term.index(te)] + "\t" + t + "\t" + text + "\n")

                    for i, token in enumerate(word_tokenize(text)):

                        if token not in ' '.join(term):
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
                    id = s[left + 4:right - 1]
                    while not "</sentence>" in s:
                        if "<text>" in s:
                            left = s.find("<text>")
                            right = s.find("</text>")
                            text = s[left + 6:right]
                        if "aspectTerm" in s and "aspectTerms" not in s:
                            left = s.find("term=")
                            right = s.find("polarity=")
                            term.append(s[left + 6:right - 2])
                            left = s.find("polarity=")
                            right = s.find("from=")
                            polarity.append(s[left + 10:right - 2])

                        s = f.readline().strip()

                    for te in term:
                        te_token = te.split(' ')
                        if len(te_token) == 1:
                            g.write(id + "\t" + 'single ' + polarity[term.index(te)] + "\t" + te + "\t" + text + "\n")
                        else:
                            for i, t in enumerate(te_token):
                                if i == 0:
                                    g.write(id + "\t" + 'first word ' + polarity[
                                        term.index(te)] + "\t" + t + "\t" + text + "\n")
                                elif i == len(te_token) - 1:
                                    g.write(
                                        id + "\t" + 'last word ' + polarity[
                                            term.index(te)] + "\t" + t + "\t" + text + "\n")
                                else:
                                    g.write(id + "\t" + 'middle word ' + polarity[
                                        term.index(te)] + "\t" + t + "\t" + text + "\n")

                    for i, token in enumerate(word_tokenize(text)):

                        if token not in ' '.join(term):
                            g.write(id + "\t" + "none" + "\t" + token + "\t" + text + "\n")


                else:
                    s = f.readline().strip()
