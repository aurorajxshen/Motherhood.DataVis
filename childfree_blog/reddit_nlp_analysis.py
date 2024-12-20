import pandas as pd
import nltk
import re
import networkx as nx
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the data
file_path = '/Users/jiaxinshen/Desktop/NLP_motherhood/reddit_comments_network_analysis.csv'  # Update with your CSV file
comments_df = pd.read_csv(file_path)

# Initialize lemmatizer and extended stopwords
lemmatizer = WordNetLemmatizer()
custom_stopwords = list(set(stopwords.words('english')).union({
    'im', 'like', 'dont', 'thats', 'want', 'just', 'people', 'children', 'kids', 'child', 'think', 'life'
}))

# Preprocessing function with lemmatization and extended stopwords
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and special characters
    tokens = text.split()  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
comments_df['cleaned_comment'] = comments_df['comment'].apply(preprocess_text)

# 1. Word Cloud
wordcloud_text = ' '.join(comments_df['cleaned_comment'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reddit Comments')
plt.show()

# 2. Sentiment Analysis
sia = SentimentIntensityAnalyzer()
comments_df['sentiment'] = comments_df['cleaned_comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Sentiment Categorization
comments_df['sentiment_label'] = comments_df['sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

# Sentiment Distribution Plot
sentiment_counts = comments_df['sentiment_label'].value_counts()
sentiment_counts.plot(kind='bar', figsize=(8, 6))
plt.title('Sentiment Distribution of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.show()

# 3. Topic Modeling using LDA with refined parameters
vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words=custom_stopwords)
X = vectorizer.fit_transform(comments_df['cleaned_comment'])

# Apply LDA for 6 topics
lda = LatentDirichletAllocation(n_components=6, random_state=42)
lda.fit(X)

# Display topics with more words
print("\nTopics Found by LDA:")
for index, topic in enumerate(lda.components_):
    print(f"Topic {index + 1}:", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-15:]])

# Visualize Topic Distribution
topic_results = lda.transform(X)
comments_df['Topic'] = topic_results.argmax(axis=1)
topic_counts = comments_df['Topic'].value_counts().sort_index()
topic_counts.plot(kind='bar', figsize=(8, 6))
plt.title('Distribution of Topics in Comments')
plt.xlabel('Topic')
plt.ylabel('Number of Comments')
plt.show()

# 4. Network Analysis
# Construct a network graph based on parent_id and comment_id
G = nx.DiGraph()

# Add edges to the graph (parent_id -> comment_id)
for _, row in comments_df.iterrows():
    if pd.notnull(row['parent_id']) and row['parent_id'] != row['comment_id']:
        G.add_edge(row['parent_id'], row['comment_id'], weight=row.get('upvotes', 1))

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Top 5 influencers based on centrality
top_influencers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 Influencers Based on Degree Centrality:")
for node, centrality in top_influencers:
    print(f"Comment ID: {node}, Centrality: {centrality:.4f}")

# Visualize the network (simplified)
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.15)
nx.draw(G, pos, node_size=20, alpha=0.6, edge_color='gray', with_labels=False)
nx.draw_networkx_nodes(G, pos, nodelist=[n for n, _ in top_influencers], node_size=100, node_color='red')
plt.title('Network Visualization of Reddit Comments')
plt.show()

# Save cleaned data for further use
comments_df.to_csv('processed_reddit_comments.csv', index=False)
