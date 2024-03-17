import os
import sys
import re
import time
import pandas as pd
import numpy as np



def lowercase_text(text):
    words = text.split()
    lower_words = [word.lower() for word in words]
    return ' '.join(lower_words)


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# Contractions mapping
contractions_map = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "i've": "i have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text



stopwords = ['a', 'an', 'the', 'is', 'are', 'and', 'or', 'but', 'for', 'in', 'on', 
             'at', 'with', 'to', 'from', 'of', 'by', 'as', 'it', 'its', 'this', 
             'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you', 'me', 'him', 
             'her', 'us', 'them', 'i', 'my', 'mine', 'your', 'yours', 'his', 'hers', 
             'their', 'theirs', 'our', 'ours', 'not', 'no', 'nor', 'so', 'too', 'very', 
             'just', 'can', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 
             'shall', 'has', 'have', 'had', 'do', 'does', 'did', 'am', 'is', 'are', 
             'was', 'were', 'be', 'being', 'been']

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


def stem(word):
    # Implement the Porter Stemmer algorithm here
    # This is a simplified example
    if word.endswith('ing'):
        return word[:-3]
    elif word.endswith('ed'):
        return word[:-2]
    else:
        return word

def stem_text(text):
    words = text.split()
    stemmed_words = [stem(word) for word in words]
    return ' '.join(stemmed_words)


# A very basic dictionary for lemmatization
lemma_dict = {'am': 'be', 'are': 'be', 'is': 'be', 'was': 'be', 'were': 'be',
              'having': 'have', 'has': 'have', 'had': 'have'}

def lemmatize(word):
    # Check if the word is in the lemma dictionary
    if word in lemma_dict:
        return lemma_dict[word]
    else:
        return word

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def remove_punctuation(text):
    #it matches any character that is not a word character or whitespace.
    return re.sub(r'[^\w\s]', ' ', text) if isinstance(text, str) else text


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_single_character(text):
    return ' '.join([w for w in text.split() if len(w) > 1 or w == 'i'] )

"""
    Split the input DataFrame into training and testing sets based on the specified train_size and test_size ratios.

    Parameters:
    df (DataFrame): The input DataFrame to be split.
    train_size (float): The proportion of the DataFrame to include in the training set. Default is 0.8.
    test_size (float): The proportion of the DataFrame to include in the testing set. Default is 0.2.

    Returns:
    tuple: A tuple containing the training DataFrame and testing DataFrame.
"""

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def train_test_split(df, train_size=0.8, test_size=0.2):
    # Ensure reproducibility
    np.random.seed(42)
    #Shuffle the DataFrame
    #df = df.sample(frac=1).reset_index(drop=True)
    train_rows = int(len(df) * train_size)
    # Calculate the number of rows to use for testing
    test_rows = int(len(df) * test_size)
    # Split the DataFrame into training and testing sets
    train_df = df[:train_rows]
    test_df = df[train_rows:train_rows + test_rows + 1]

    return train_df, test_df


def extract_keywords_by_tf_idf():
    pass


def main():
    # python CS585_P02_AXXXXXXXX.py TRAIN_SIZE
    if len(sys.argv) != 2:
        print("Warning: using default values TRAIN_SIZE=80.")
        train_size = 0.8
    else:    
        _, train_size = sys.argv
        if is_number(train_size):
            train_size = float(train_size)/100
        else:
            print("Warning: using default values TRAIN_SIZE=80.")
            train_size = 0.8
        if train_size <20 or train_size > 80:
            print("Warning: using default values TRAIN_SIZE=80.")
            train_size = 0.8

    dataset_file = "Reviews.csv"
    
    #FILdENAME is the input CSV file name (graph G data)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = cur_dir + '/' + dataset_file
    df = pd.read_csv(csv_file_path, usecols=['Id', 'Score', 'Summary', 'Text'], header=0)

    print(f'number of samples: {len(df)}')

    unique_titles = df['Text'].unique()
    print(f'number of unique Text: {len(unique_titles)}')

    unique_labels = df['Score'].unique()
    print(f'unique_labels: {unique_labels}')

    #------------------------clean up the dataset --------------------------------------
    #remove the duplicate rows(texts) in dataset
    print("remove duplicates in dataset")
    df = df.drop_duplicates(subset=['Text'], keep='first')

    unique_texts = df['Text']
    print(f'number of unique texts: {len(unique_texts)}')

    print(df.head())

    start_time = time.time()
    
    df['Text'] = df['Text'].apply(lambda x: lowercase_text(x))
    df['Text'] = df['Text'].apply(lambda x: remove_html_tags(x))
    df['Text'] = df['Text'].apply(lambda x: expand_contractions(x, contractions_map))
    df['Text'] = df['Text'].apply(lambda x: remove_punctuation(x))
    df['Text'] = df['Text'].apply(lambda x: remove_stopwords(x))
    df['Text'] = df['Text'].apply(lambda x: stem_text(x))
    df['Text'] = df['Text'].apply(lambda x: lemmatize_text(x))
    df['Text'] = df['Text'].apply(lambda x: remove_numbers(x))
    df['Text'] = df['Text'].apply(lambda x: remove_single_character(x))
    
    #convert the Score 1 ～ 5 to good and bad values
    df['Label'] = df['Score'].apply(lambda x: 'good' if x > 3 else 'bad')
    end_time = time.time()
    print("---  clean up data in seconds --- %s" % (end_time - start_time ))
    

    print(df.head())

    #---------------------------Bayes  Classifier network learning ---------------------------------

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, train_size=train_size, test_size=1-train_size)
    print(f"Train Rows: {len(train_df)}, Test Rows: {len(test_df)}")
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    #P(Label=good) and P(Label=bad
    count_label_good = len(train_df[train_df['Label'] == 'good'])
    count_label_bad = len(train_df[train_df['Label'] == 'bad'])
    prob_label_good =  count_label_good / len(train_df)
    prob_label_bad =  count_label_bad / len(train_df)


    print(f'P(Label=good): {prob_label_good}')
    print(f'P(Label=bad): {prob_label_bad}')


    #build the vocabulary from the dataset
    vocab_in_training = set()
    for text in train_df['Text']:
        vocab_in_training.update(text.split())

    print(f'Vocabulary size in training dataset: {len(vocab_in_training)}')

    

    #Learn every word over the vocabulary probability with label = good and label = bad 
    smoothing_alpha = 1 #avoiding zero probability
    prob_word_given_label_good_dict = {}
    prob_word_given_label_bad_dict = {}
    count_word = 100

    print("\n###### Bayes classifier network training ......\n")
    for word in vocab_in_training:
        prob_word_given_label_good = len(train_df[(train_df['Label'] == 'good') & (train_df['Text'].str.contains(word))]) + smoothing_alpha\
                                      / (count_label_good  + smoothing_alpha * count_label_good)
        
        prob_word_given_label_bad = len(train_df[(train_df['Label'] == 'bad') & (train_df['Text'].str.contains(word))]) + smoothing_alpha\
                                      / (count_label_bad  + smoothing_alpha * count_label_bad)

        prob_word_given_label_good_dict[word] = prob_word_given_label_good
        prob_word_given_label_bad_dict[word] = prob_word_given_label_bad
        if count_word > 0:
            count_word -= 1
            print(f'P({word}|Label=good): {prob_word_given_label_good}')
            print(f'P({word}|Label=bad): {prob_word_given_label_bad}')


    #---------------------------Bayes  Classifier network prediction  ---------------------------------
    print("\n###### Bayes classifier network test ......\n")
    for review, label in test_df['Text', 'Label'].iterrows():
        predict_prob_label_good = prob_label_good
        predict_prob_label_bad = prob_label_bad
        for word in review.split():
            predict_prob_label_good *=  prob_word_given_label_good_dict.get(word, 1.0)
            predict_prob_label_bad *=  prob_word_given_label_bad_dict.get(word, 1.0)
        
        if predict_prob_label_good > predict_prob_label_bad:
            print(f'Predicted Label: good, Actual Label: {label}')
        else:
            print(f'Predicted Label: bad, Actual Label: {label}')

if __name__ == "__main__":
    main()
