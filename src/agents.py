from image_analysis import object_identification, color_identification, position_extraction, character_recognition
from text_analysis import text_summarization, key_phrase_identification, narrative_understanding

class ImageAnalysisAgent:
    def analyze(self, image_path):
        objects = object_identification(image_path)
        colors = color_identification(image_path)
        positions = position_extraction(image_path)
        text = character_recognition(image_path)
        return {
            'objects': objects,
            'colors': colors,
            'positions': positions,
            'text': text
        }

class TextAnalysisAgent:
    def analyze(self, text):
        summary = text_summarization(text)
        key_phrases = key_phrase_identification(text)
        narrative = narrative_understanding(text)
        return {
            'summary': summary,
            'key_phrases': key_phrases,
            'narrative': narrative
        }
