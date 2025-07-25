import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


stopwords = {
    'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been',
    'has', 'have', 'had',
    'do', 'does', 'did',
    'will', 'shall', 'would', 'should',
    'can', 'could', 'may', 'might', 'must',
    'the', 'a', 'an', 'to', 'of', 'in', 'on', 'for', 'at', 'by',
    'with', 'and', 'or', 'as', 'but', 'if', 'so', 'because'
}

VIDEO_DIR = "static/videos/"

def process_text(text):
    tokens = word_tokenize(text.lower())  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  
    tokens = [word for word in tokens if word not in stopwords]  

    video_sequence = []

    for word in tokens:
        video_path = os.path.join(VIDEO_DIR, f"{word}.mp4")
        if os.path.exists(video_path):
            video_sequence.append(f"{word}.mp4")
        else:
           
            for char in word:
                char_path = os.path.join(VIDEO_DIR, f"{char}.mp4")
                if os.path.exists(char_path):
                    video_sequence.append(f"{char}.mp4")

    return video_sequence
