import spacy
nlp = spacy.load('en_core_web_sm') 

text = "The release - 4 is better than release - 2. The latest is the release - 6."
my_doc = nlp(text)

# Implement Token Matcher
from spacy.matcher import Matcher   # Import Matcher from spacy
matcher = Matcher(nlp.vocab)  # Initialize Matcher object

# Define the pattern to identify phrases like release - 4 and so on
my_pattern = [{"LOWER": "release"}, {"IS_PUNCT": True}, {"LIKE_NUM": True}]
matcher.add('Release', None, my_pattern) # Define the token matcher

matches = matcher(my_doc)  # Return a list of tuples with structure : (match_id, start, end)
#print(matches)

# Extract the matches
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = my_doc[start:end]
    #print(span.text)

# Example 1: Extract phrases mentioning various places

text1 = """I visited Bangalore last time. I want to visit ladakh this summer
    I also want to visit Gangtok and I am planning to visit Kedarnath."""
doc = nlp(text1)
# My desired pattern consist of 2 tokens. One is visited/visit for which
# we use LEMMA to get the root word and other token is place which 
# we can get using POS tag as "PROPN"

match = Matcher(nlp.vocab)  # Initialize Matcher
pattern = [{"LEMMA": "visit"}, {"POS": "PROPN"}] # Define the my_pattern
match.add("Places", None, pattern)
matching = match(doc)

for match_id, start, end in matching:
    print(doc[start:end].text)
