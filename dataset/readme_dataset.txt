Employed a 3-term wide sliding window approach in extracting features

- previous term
- the term to be classified
- next term

features for each term
- Start Capital (whether the term is capitalized or not), 
- All Capital (If the term is all uppercase),
- Capital Ratio (the ratio of capital letters in the term) 
- Length (number of characters in the word), 
- Vowel Ratio (The ratio of number consonant over the number of vowels in the word) 
- Number of Vowels (The number vowels in the word) 
- Number of numeric characters (The number of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)
- Numeric Ratio (The ratio of numerical characters {0,1,2,3,4,5,6,7,8,9} in the term)
- Number Of NonAlphanumeric  (The number of characters except alphanumerical characters a-z and 0-9 in the term) 
- NonAlphanumericRatio (The ratio of characters except alphanumerical characters a-z and 0-9 in the term) 