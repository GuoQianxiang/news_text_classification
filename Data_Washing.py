#   清洗数据
# training_data = re.sub(r'\d+|\W+', ' ', training_data)   # 移除数字和标点符号 # 去除无关字符：这可能包括去除标点符号、数字、特殊字符等。可以使用Python的str.replace()函数或正则表达式库re来实现。(把关键数字列删掉了)
# print(training_data)
# training_data = training_data.lower()     #转换为小写：为了保证一致性，通常会将所有文本转换为小写。这可以使用str.lower()函数来实现
# # print(training_data)
# training_data = training_data.split()    # 分词：将文本分解为单独的词语。这可以使用Python的str.split()函数或自然语言处理库如NLTK来实现
# # print(training_data)
# stop_words = set(stopwords.words('english'))    # 去除停用词：停用词是指在文本中频繁出现但对文本含义贡献不大的词，如“the”、“is”、“and”等。可以使用NLTK库中的停用词列表来移除这些词。
# training_data = [word for word in training_data if word not in stop_words]  #会不会数字和报刊等数据也被处理掉了
# # stemmer = PorterStemmer()
# # words = [stemmer.stem(word) for word in words]  #词干提取或词形还原：这是将词语转换为其基本形式的过程。例如，“running”转换为“run”。可以使用NLTK或spaCy等库来实现。