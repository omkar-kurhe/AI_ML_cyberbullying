import pandas as pd
import re
import unicodedata
from collections import Counter
from textblob import TextBlob

# Load the dataset
data = pd.read_csv('labeled_data.csv')


# Clean the text data
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip spaces
    return text


data['cleaned_tweet'] = data['tweet'].apply(clean_text)

# Map class labels to meaningful names (0: hate speech, 1: offensive, 2: neutral)
class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neutral'}
data['class_label'] = data['class'].map(class_mapping)


# Bully emoji detection
def detect_bully_emojis(text):
    bully_emojis = {"😏": -0.5, "😡": -1, "😠": -1, "👿": -1, "🖕": -1.5, "🙍": -0.8, "😭": -0.7, "🔪": -1.5, "🔫": -1.5,
                    "🔨": -1, "⚒": -1, "🪓": -1, "🥵": -0.5, "🤬": -1.5, "👺": -1, "👹": -1, "😈": -1, "☠": -1.5}
    text = unicodedata.normalize("NFC", text)  # Normalize Unicode text
    found_emojis = [emoji for emoji in bully_emojis if emoji in text]
    sentiment_adjustment = sum(bully_emojis[emoji] for emoji in found_emojis)
    return found_emojis, sentiment_adjustment


# Sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Determine severity level
def determine_severity(sentiment_score, emoji_adjustment):
    total_score = sentiment_score + emoji_adjustment
    if total_score < -1.5:
        return "High"
    elif total_score < -0.5:
        return "Medium"
    else:
        return "Low"


# Function to check if a sentence is bullying or not
def is_bullying_sentence(sentence):
    sentence = clean_text(sentence)  # Clean the input sentence
    matches = data[data['cleaned_tweet'] == sentence]  # Check for exact match
    if not matches.empty:
        majority_class = matches['class_label'].mode()[0]
        return f"The sentence '{sentence}' is detected as {majority_class}."
    return f"The sentence '{sentence}' is not detected as bullying."


# Log detected bullying content
def log_bullying(sentence, severity):
    with open("bullying_log.txt", "a") as log_file:
        log_file.write(f"Sentence: {sentence} | Severity: {severity}\n")


# Store past sentences for analysis
sentence_history = []

# Loop to keep checking sentences
while True:
    sentence_to_check = input("\nEnter a sentence to check if it is bullying or not: ")
    emoji_detected, emoji_sentiment = detect_bully_emojis(sentence_to_check)
    text_sentiment = analyze_sentiment(sentence_to_check)
    severity = determine_severity(text_sentiment, emoji_sentiment)
    text_result = is_bullying_sentence(sentence_to_check)

    if emoji_detected:
        print("Bully Emojis Found:", emoji_detected)

    print(text_result)
    print(f"Sentiment Score: {text_sentiment:.2f}, Adjusted by Emojis: {emoji_sentiment:.2f}")
    print(f"Severity Level: {severity}")

    if severity in ["Medium", "High"]:
        log_bullying(sentence_to_check, severity)

    sentence_history.append(sentence_to_check)

    if len(sentence_history) % 3 == 0:
        common_words = Counter(" ".join(sentence_history).split()).most_common(5)
        print("\nTrending bullying-related words:", common_words)

    choice = input("\nPress 1 to check another sentence, or 0 to exit: ")
    if choice == "0":
        print("Exiting the program.")
        break
