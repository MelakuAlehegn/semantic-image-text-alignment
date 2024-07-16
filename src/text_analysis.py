from transformers import pipeline

summarizer = pipeline('summarization')
ner = pipeline('ner', grouped_entities=True)

def text_summarization(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def key_phrase_identification(text):
    entities = ner(text)
    return [entity['word'] for entity in entities]

def narrative_understanding(text):
    # Custom logic for narrative understanding
    # For simplicity, combining summarization and key phrases
    summary = text_summarization(text)
    key_phrases = key_phrase_identification(text)
    return summary, key_phrases
