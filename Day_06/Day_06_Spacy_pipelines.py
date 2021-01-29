## Spacy Pipline Operations and Examples ##
import spacy
nlp = spacy.load('en_core_web_sm')
print("Initial Pipeline : ",nlp.pipe_names)            # Inspect the pipeline
print("Checking if it has textcat : ",nlp.has_pipe('textcat'))     # Check if pipeline component is present
nlp.add_pipe(nlp.create_pipe('textcat'), before='ner')  # Adding new pipeline component and where to add
print("Pipeline components after adding textcat: ",nlp.pipe_names)
# Removing the pipeline components
nlp.remove_pipe("textcat")
print("Pipeline Components after removing textcat: ",nlp.pipe_names)
# Disable the components of a pipeline
nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser"])
print("Disabling the components: ",nlp.pipe_names)
# We can also create custom pipeline components
print("Creating the custom pipline component")
def my_component(doc):
    named_ent = [token.label_ for token in doc.ents]
    print(named_ent)
    return doc
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(my_component, after='ner')
print(nlp.pipe_names)
doc = nlp("India is a democratic country")