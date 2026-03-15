# stanza_api_proxy.py
import requests

class StanzaAPIProxy:
    """
    A drop-in replacement that mimics stanza.Pipeline but calls your API
    """
    
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self._check_connection()
    
    def _check_connection(self):
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code != 200:
                print(f"Warning: Stanza API not responding properly at {self.api_url}")
        except:
            print(f"Warning: Could not connect to Stanza API at {self.api_url}")
            print("Make sure the API server is running with: python stanza_api.py")
    
    def __call__(self, text: str):
        """
        This makes the object callable like nlp("text")
        Returns a Document-like object
        """
        return self._process_text(text)
    
    def _process_text(self, text: str):
        """Call the API and reconstruct a document-like object"""
        try:
            response = requests.post(
                f"{self.api_url}/process",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Reconstruct a document-like object
            return self._reconstruct_document(data)
            
        except Exception as e:
            print(f"Error calling Stanza API: {e}")
            # Return an empty document-like object
            return self._empty_document(text)
    
    def _reconstruct_document(self, data):
        """Reconstruct a document with the same interface as stanza.Document"""
        
        # Create simple classes that mimic Stanza's structure
        class Word:
            def __init__(self, word_data):
                self.id = word_data['id']
                self.text = word_data['text']
                self.lemma = word_data['lemma']
                self.upos = word_data['upos']
                self.xpos = word_data['xpos']
                self.feats = word_data['feats']
                self.head = word_data['head']
                self.deprel = word_data['deprel']
        
        class Sentence:
            def __init__(self, sent_data):
                self.words = [Word(w) for w in sent_data['words']]
                self.text = sent_data['text']
        
        class Document:
            def __init__(self, doc_data):
                self.text = doc_data['text']
                self.sentences = [Sentence(s) for s in doc_data['sentences']]
                self.num_sentences = doc_data['num_sentences']
        
        return Document(data)
    
    def _empty_document(self, text):
        """Return an empty document structure when API fails"""
        class EmptyDocument:
            def __init__(self, text):
                self.text = text
                self.sentences = []
                self.num_sentences = 0
        
        return EmptyDocument(text)

# For convenience, also create a Pipeline function that mimics stanza.Pipeline
def Pipeline(lang='fr', processors='tokenize,pos,lemma,depparse', api_url="http://localhost:5000"):
    """
    This mimics stanza.Pipeline but returns our proxy
    """
    print(f"Using Stanza API proxy (connecting to {api_url})")
    return StanzaAPIProxy(api_url)
