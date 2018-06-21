import pandas as pd
import openpyxl


document_token_new = document_token.copy()

##ADDING NEW FEATURES TO DATASET

#Start Capital (whether the term is capitalized or not)
document_token_new['start_capital']= 0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    if word[0].isupper():
        document_token_new.at[index,'start_capital'] = 1

#All Capital (If the term is all uppercase)
document_token_new['all_capital']= 0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    if word.isupper():
        document_token_new.at[index,'all_capital'] = 1

#Capital Ratio (the ratio of capital letters in the term)
document_token_new['capital_ratio']= 0.0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    ratio = (sum(1 for c in word if c.isupper())) / len(word)
    document_token_new.at[index,'capital_ratio'] = ratio
    
#Length (number of characters in the word)
document_token_new['length']= 0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    document_token_new.at[index,'length'] = len(word)

#Vowel Ratio (The ratio of number consonant over the number of vowels in the word)
#Number of Vowels (The number vowels in the word) 
document_token_new['vowel_ratio']= 0.0
document_token_new['number_of_vowels']= 0.0
vowels = list("aeıioöuü")
consonants = list("bcçdfgğhjklmnpqrsştvwyxz") 
for index, row in document_token_new.iterrows():
    word = str(row['token_text']).lower()
    
    number_of_consonants = sum(word.count(c) for c in consonants)
    number_of_vowels = sum(word.count(c) for c in vowels)
    
    try:
        vowel_ratio = number_of_consonants / number_of_vowels
    except:
        vowel_ratio = number_of_consonants
    
    document_token_new.at[index,'vowel_ratio'] = vowel_ratio
    document_token_new.at[index,'number_of_vowels'] = number_of_vowels

#Number of numeric characters (The number of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)
#Numeric Ratio (The ratio of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)    
document_token_new['numeric_charecter']= 0.0
document_token_new['numeric_ratio']= 0.0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    number_numeric_charecter = sum(1 for c in word if c.isnumeric())
    document_token_new.at[index,'numeric_charecter'] = number_numeric_charecter
    document_token_new.at[index,'numeric_ratio'] = number_numeric_charecter / len(word)

#Number Of NonAlphanumeric  (The number of characters except alphanumerical characters a-z and 0-9 in the term) 
#NonAlphanumericRatio (The ratio of characters except alphanumerical characters a-z and 0-9 in the term) 
document_token_new['number_of_nonAlphanumeric']= 0.0
#document_token_new['nonAlphanumericRatio ']= 0.0
for index, row in document_token_new.iterrows():
    word = str(row['token_text'])
    nonAlphanumberic_word = ''.join([i for i in word if not i.isalnum()])
    
    document_token_new.at[index,'number_of_nonAlphanumeric'] = len(nonAlphanumberic_word)
    document_token_new.at[index,'nonAlphanumericRatio'] = len(nonAlphanumberic_word) / len(word)

error_row = []
for index, row in document_token_new.iterrows():
    if type(row['c_id']) is type(None):
        error_row.append(index)

        
document_token_new = document_token_new.drop(document_token_new.index[error_row])


############################################################################################
doc_token_prev = document_token_new.copy()
doc_token_prev['previous_word_text'] = ''
for index, row in document_token_new.iterrows():
    doc_id = int(row['doc_id'])
    token_id = int(row['token_id']) 
    query = 'doc_id == ' + str(doc_id) + ' & token_id == ' + str(token_id-1)
    response = document_token_new.query(query)
    try:
        for i,r in response.iterrows():
            doc_token_prev.at[index,'previous_word_text'] = str(r['token_text'])
    except:
        pass


doc_token_prev['prev_start_capital']= 0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    try:
        if word[0].isupper():
            doc_token_prev.at[index,'prev_start_capital'] = 1
    except:
        pass


doc_token_prev['prev_all_capital']= 0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    try:
        if word.isupper():
            doc_token_prev.at[index,'prev_all_capital'] = 1
    except:
        pass

doc_token_prev['prev_capital_ratio']= 0.0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    try:
        ratio = (sum(1 for c in word if c.isupper())) / len(word)
        doc_token_prev.at[index,'prev_capital_ratio'] = ratio
    except:
        pass

doc_token_prev['prev_length']= 0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    doc_token_prev.at[index,'prev_length'] = len(word)


doc_token_prev['prev_vowel_ratio']= 0.0
doc_token_prev['prev_number_of_vowels']= 0.0
vowels = list("aeıioöuü")
consonants = list("bcçdfgğhjklmnpqrsştvwyxz") 
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text']).lower()
    
    number_of_consonants = sum(word.count(c) for c in consonants)
    number_of_vowels = sum(word.count(c) for c in vowels)
    
    try:
        vowel_ratio = number_of_consonants / number_of_vowels
    except:
        vowel_ratio = number_of_consonants
    
    doc_token_prev.at[index,'prev_vowel_ratio'] = vowel_ratio
    doc_token_prev.at[index,'prev_number_of_vowels'] = number_of_vowels

#Number of numeric characters (The number of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)
#Numeric Ratio (The ratio of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)    
doc_token_prev['prev_numeric_charecter']= 0.0
doc_token_prev['prev_numeric_ratio']= 0.0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    try:
        number_numeric_charecter = sum(1 for c in word if c.isnumeric())
        doc_token_prev.at[index,'prev_numeric_charecter'] = number_numeric_charecter
        doc_token_prev.at[index,'prev_numeric_ratio'] = number_numeric_charecter / len(word)
    except:
        pass
    
#Number Of NonAlphanumeric  (The number of characters except alphanumerical characters a-z and 0-9 in the term) 
#NonAlphanumericRatio (The ratio of characters except alphanumerical characters a-z and 0-9 in the term) 
doc_token_prev['prev_number_of_nonAlphanumeric']= 0.0
#doc_token_prev['prev_nonAlphanumericRatio']= 0.0
for index, row in doc_token_prev.iterrows():
    word = str(row['previous_word_text'])
    try:
        nonAlphanumberic_word = ''.join([i for i in word if not i.isalnum()])
        
        doc_token_prev.at[index,'prev_number_of_nonAlphanumeric'] = len(nonAlphanumberic_word)
        doc_token_prev.at[index,'prev_nonAlphanumericRatio'] = len(nonAlphanumberic_word) / len(word) 
    except:
        pass
##########################################################################################
doc_token_next = doc_token_prev.copy()
doc_token_next['next_word_text'] = ''
for index, row in document_token_new.iterrows():
    doc_id = int(row['doc_id'])
    token_id = int(row['token_id']) 
    query = 'doc_id == ' + str(doc_id) + ' & token_id == ' + str(token_id+1)
    response = document_token_new.query(query)
    try:
        for i,r in response.iterrows():
            doc_token_next.at[index,'next_word_text'] = str(r['token_text'])
    except:
        pass
    
    
doc_token_next['next_start_capital']= 0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    try:
        if word[0].isupper():
            doc_token_next.at[index,'next_start_capital'] = 1
    except:
        pass


doc_token_next['next_all_capital']= 0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    try:
        if word.isupper():
            doc_token_next.at[index,'next_all_capital'] = 1
    except:
        pass

doc_token_next['next_capital_ratio']= 0.0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    try:
        ratio = (sum(1 for c in word if c.isupper())) / len(word)
        doc_token_next.at[index,'next_capital_ratio'] = ratio
    except:
        pass

doc_token_next['next_length']= 0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    doc_token_next.at[index,'next_length'] = len(word)


doc_token_next['next_vowel_ratio']= 0.0
doc_token_next['next_number_of_vowels']= 0.0
vowels = list("aeıioöuü")
consonants = list("bcçdfgğhjklmnpqrsştvwyxz") 
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text']).lower()
    
    number_of_consonants = sum(word.count(c) for c in consonants)
    number_of_vowels = sum(word.count(c) for c in vowels)
    
    try:
        vowel_ratio = number_of_consonants / number_of_vowels
    except:
        vowel_ratio = number_of_consonants
    
    doc_token_next.at[index,'next_vowel_ratio'] = vowel_ratio
    doc_token_next.at[index,'next_number_of_vowels'] = number_of_vowels

#Number of numeric characters (The number of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)
#Numeric Ratio (The ratio of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)    
doc_token_next['next_numeric_charecter']= 0.0
doc_token_next['next_numeric_ratio']= 0.0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    try:
        number_numeric_charecter = sum(1 for c in word if c.isnumeric())
        doc_token_next.at[index,'next_numeric_charecter'] = number_numeric_charecter
        doc_token_next.at[index,'next_numeric_ratio'] = number_numeric_charecter / len(word)
    except:
        pass
    
#Number Of NonAlphanumeric  (The number of characters except alphanumerical characters a-z and 0-9 in the term) 
#NonAlphanumericRatio (The ratio of characters except alphanumerical characters a-z and 0-9 in the term) 
doc_token_next['next_number_of_nonAlphanumeric']= 0.0
#doc_token_next['next_nonAlphanumericRatio ']= 0.0
for index, row in doc_token_next.iterrows():
    word = str(row['next_word_text'])
    try:
        nonAlphanumberic_word = ''.join([i for i in word if not i.isalnum()])
        
        doc_token_next.at[index,'next_number_of_nonAlphanumeric'] = len(nonAlphanumberic_word)
        doc_token_next.at[index,'next_nonAlphanumericRatio'] = len(nonAlphanumberic_word) / len(word) 
    except:
        pass    
##########################################################################################
dataframe = doc_token_next.copy()
dataframe['class'] = dataframe['c_id']
dataframe.__delitem__('c_id')
dataframe.__delitem__('doc_id')
dataframe.__delitem__('token_id')

#dataframe.__delitem__('class')
prev_word_features = ['prev_start_capital','prev_all_capital','prev_capital_ratio','prev_length','prev_vowel_ratio',
                      'prev_number_of_vowels','prev_numeric_charecter','prev_numeric_ratio','prev_number_of_nonAlphanumeric',
                      'prev_nonAlphanumericRatio']
next_word_features = ['next_start_capital','next_all_capital','next_capital_ratio','next_length','next_vowel_ratio',
                      'next_number_of_vowels','next_numeric_charecter','next_numeric_ratio','next_number_of_nonAlphanumeric',
                      'next_nonAlphanumericRatio']

#normalization
for index, row in dataframe.iterrows():
    if row['previous_word_text'] == '':
        for feature in prev_word_features:
            dataframe.at[index,feature] = dataframe[feature].mean()
    
    if row['next_word_text'] == '':
        for feature in next_word_features:
            dataframe.at[index,feature] = dataframe[feature].mean()    
    
   
    
    
#########################################################################################3    
'''
dataframe.to_csv('presentation/alldataset.csv',index=False)
writer = pd.ExcelWriter('presentation/alldataset.xlsx')
dataframe.to_excel(writer)
writer.save()

dataframe = dataframe[dataframe['class'] != 11]

dataframe_y = dataframe['class']
dataframe.__delitem__('class')
dataframe_y = pd.DataFrame(dataframe_y)
writer = pd.ExcelWriter('presentation/y.xlsx')
dataframe_y.to_excel(writer)
writer.save()

dataframe.__delitem__('token_text')
dataframe.__delitem__('previous_word_text')
dataframe.__delitem__('next_word_text')
dataframe_x = dataframe
writer = pd.ExcelWriter('presentation/X.xlsx')
dataframe_x.to_excel(writer)
writer.save()
'''
