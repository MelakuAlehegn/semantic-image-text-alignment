import unittest
from src.agents import ImageAnalysisAgent, TextAnalysisAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.image_agent = ImageAnalysisAgent()
        self.text_agent = TextAnalysisAgent()

    def test_image_analysis(self):
        image_path = '../sample_data/fbb156a0dc35c2e9f126d5140ae6ea66/endframe_1.jpg'
        result = self.image_agent.analyze(image_path)
        self.assertIn('objects', result)
        self.assertIn('colors', result)
        self.assertIn('positions', result)
        self.assertIn('text', result)

    def test_text_analysis(self):
        text = "This is a sample text description for the advertisement."
        result = self.text_agent.analyze(text)
        self.assertIn('summary', result)
        self.assertIn('key_phrases', result)
        self.assertIn('narrative', result)

if __name__ == '__main__':
    unittest.main()
