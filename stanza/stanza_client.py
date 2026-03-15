# client.py
import requests

class StanzaClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def process(self, text):
        """Process single text"""
        response = requests.post(
            f"{self.base_url}/process",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    def process_batch(self, texts):
        """Process multiple texts"""
        response = requests.post(
            f"{self.base_url}/process_batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        return response.json()

# Usage example
if __name__ == "__main__":
    client = StanzaClient()
    
    # Check health
    print("Health check:", client.health_check())
    
    # Process single text
    text = "Bonjour, comment allez-vous? Je m'appelle Jean."
    result = client.process(text)
    
    print(f"\nProcessed text: {result['text']}")
    print(f"Number of sentences: {result['num_sentences']}")
    
    for i, sent in enumerate(result['sentences'], 1):
        print(f"\nSentence {i}: {sent['text']}")
        for word in sent['words'][:3]:  # Show first 3 words
            print(f"  Word: {word['text']:<10} POS: {word['upos']:<8} Lemma: {word['lemma']}")
    
    # Process batch
    texts = [
        "Premier texte à analyser.",
        "Deuxième exemple avec plus de mots."
    ]
    batch_result = client.process_batch(texts)
    print(f"\nBatch processed {batch_result['num_texts']} texts")
