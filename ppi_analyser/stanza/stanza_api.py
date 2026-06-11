# stanza_api.py
import importlib
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StanzaNLP:
    def __init__(self, language='fr', processors='tokenize,pos,lemma,depparse'):
        """Initialize Stanza pipeline once"""
        logger.info(f"Loading Stanza pipeline for {language}...")
        stanza = importlib.import_module('stanza')
        self.nlp = stanza.Pipeline(
            language, 
            processors=processors,
            verbose=False  # Reduce console output
        )
        logger.info("Stanza pipeline loaded successfully!")
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and return structured results"""
        doc = self.nlp(text)
        
        # Build response with relevant annotations
        sentences_data = []
        for sent in doc.sentences:
            words_data = []
            for word in sent.words:
                word_info = {
                    'id': word.id,
                    'text': word.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'feats': word.feats,
                    'head': word.head,
                    'deprel': word.deprel,
                    # Add any other attributes you need
                }
                words_data.append(word_info)
            
            sentences_data.append({
                'words': words_data,
                'text': ' '.join([w.text for w in sent.words])
            })
        
        return {
            'text': text,
            'num_sentences': len(doc.sentences),
            'sentences': sentences_data
        }

# Initialize Flask app and Stanza
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Stanza once (when the server starts)
nlp_processor = StanzaNLP()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Stanza French NLP API',
        'language': 'fr',
        'processors': 'tokenize,pos,lemma,depparse'
    })

@app.route('/process', methods=['POST'])
def process_text():
    """Process text with Stanza"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        # Process with Stanza
        result = nlp_processor.process_text(text)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_batch', methods=['POST'])
def process_batch():
    """Process multiple texts in batch"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be an array'}), 400
        
        # Process each text
        results = []
        for text in texts:
            results.append(nlp_processor.process_text(text))
        
        return jsonify({
            'num_texts': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
