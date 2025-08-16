import csv
import numpy as np
import pandas as pd

# reading in and preprocessing the train dataset

train_dataset = []

with open("C:/Users/katri/Downloads/bbc_newstopics/bbc_train.csv", encoding="utf-8") as f:
    rd = csv.reader(f)
    for topic, text in rd:
        words = [w.lower() for w in text.split() if len(w) > 3] # removes tokens < 3 characters and lowercases all tokens
        article = [topic, words]
        train_dataset.append(article)

print(train_dataset[0])

# converting into pandas dataframe for better readability

df_train_dataset = pd.DataFrame(train_dataset)
df_train_dataset.head(5)

topics = ['tech', 'politics', 'business', 'entertainment', 'sport']

# creating unique vocabulary from the train dataset

vocabulary = []

for article in train_dataset:
    words_list = article[1]
    for word in words_list:
        vocabulary.append(word)

unique_vocabulary = list(set(vocabulary)) # converting list into set removes the duplicates
        
print(unique_vocabulary[0:10])

length_vocabulary = len(vocabulary)
print (f"Total vocabulary: {length_vocabulary}")

length_unique_vocabulary = len(unique_vocabulary)
print (f"Unique vocabulary (V): {length_unique_vocabulary}")


# calculating a probability that the word w is present in articles with the topic is c, 
# according to the formula P(w|c)= (N_w,c+1)/N_c+|V|), where:
# N_w,c (occurrencies) - number of occurrences of word w in articles with the topic c
# N_c (total_count) - total number of words in articles with the topic c
# |V| (length_unique_vocabulary) - number of unique words over all articles and topics

# occurrencies of each unique word in the articles with each of the topics

occurrencies = []

for topic in topics:
    for word in unique_vocabulary:
        count = 0
        for article in train_dataset:
            if article[0] == topic:
                words_list = article[1]
                frequency = words_list.count(word) # counts unique words
                count += frequency
        row = [topic, word, count]
        occurrencies.append(row)

print(occurrencies[0:10])

# presenting occurrencies as pandas dataframe for better readability

df_occ = pd.DataFrame(occurrencies, columns=['Topic', 'Word', 'Occurrencies'])
df_occ.head(5)

# total count of words in the articles with each of the topics

total_count = []

for topic in topics:
    count = 0
    for article in train_dataset:
        if article[0] == topic:
            words_list = article[1]
            number_words = len(words_list) # count of words in the article
            count += number_words
    row = [topic, count]
    total_count.append(row)

print(total_count)

# converting the list of word counts into a dictionary for easier access

dict_count = dict(total_count)
print(dict_count)

# calculating probabilities for each word

p = []
for index, row in df_occ.iterrows():
    p.append((row['Occurrencies'] + 1)/(dict_count.get(row['Topic']) + length_unique_vocabulary))
df_occ['Probabilities'] = p
df_occ.head(5)

# since we will operate with logs of probabilities, we can caclulate and add them to the df now

log_p = []
for index, row in df_occ.iterrows():
    log_p.append(np.log(row['Probabilities']))
df_occ['Log Probabilities'] = log_p
df_occ.head(5)

# calculating number of articles with each topic

count_articles = []
for topic in topics:
    count = 0
    for article in train_dataset:
        if article[0] == topic:
            count+=1
    row = [topic, count]
    count_articles.append(row)
print(count_articles)

# calculating probabilitises that the topic of the article is c (P(c)) and calculating logs of probabilities

p_c_list = []
for i in count_articles:
    p_c = np.log(i[1]/len(train_dataset))
    row = [i[0], p_c]
    p_c_list.append(row)
print(p_c_list)

# converting list of probabilitises into dictionary for easier access

p_c_dict = dict(p_c_list)
print(p_c_dict)

# reading in and preprocessing the test dataset

test_dataset = []

with open("C:/Users/katri/Downloads/bbc_newstopics/bbc_test.csv", encoding="utf-8") as f:
    rd = csv.reader(f)
    for topic, text in rd:
        words = [w.lower() for w in text.split() if len(w) > 3] # removes tokens < 3 characters and lowercases all tokens
        article = [topic, words]
        test_dataset.append(article)
print(test_dataset[0])

# converting test dataset into pandas dataframe for better readability

df_test_dataset = pd.DataFrame(test_dataset)
df_test_dataset.head(5)

# predicting the article topic

predicted_probabilities = []

for article in test_dataset:
    probabilities = []
    words_list = article[1]
    for key in p_c_dict:
        topic = key
        sum_log_p = 0 # initiates the sum of word probabilities log(P(w|c)) for each article
        for word in words_list:
        # finds the matching row in the dataframe with word occurrencies - there will be only one for each topic
            matching_row = df_occ[(df_occ['Word'] == word) & (df_occ['Topic'] == topic)]
            if not matching_row.empty:
                # if the matching row is found, calculates the probability
                probability = matching_row['Log Probabilities'].values[0]
                sum_log_p += probability
            else:
                # if there is no matching row, i. e. the word from the test dataset is not present in the train dataset,
                # we use the formula 1/(Nc+|V|)
                probability = np.log(1/(dict_count.get(topic) + length_unique_vocabulary))
                sum_log_p += probability
        # calculates log of probability that the topic of the article is c
        log_h_c = [topic, (p_c_dict.get(topic) + sum_log_p)]
        probabilities.append(log_h_c) # probabilities of all topics for each article
    dict_probabilities = dict(probabilities)
    max_prob = max(zip(dict_probabilities.values(), dict_probabilities.keys()))[1] # finds the most probable topic
    predicted_probabilities.append([article[1], article[0], max_prob])
print(predicted_probabilities)

df_predicted_topics = pd.DataFrame(predicted_probabilities, columns = ['Article', 'Topic', 'Predicted Topic'])
df_predicted_topics.head(10)


df_predicted_topics['Accuracy'] = df_predicted_topics['Topic'] == (df_predicted_topics['Predicted Topic'])
df_predicted_topics.head(10)

# calculates the error rate of the model

num_false = (df_predicted_topics['Accuracy'] == False).sum()
print (f"Error rate: {num_false}%")
