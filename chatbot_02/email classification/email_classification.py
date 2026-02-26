# import dependencies
import io 
import os
import pandas as pd
from openai import OpenAI 
from dotenv import load_dotenv 
from pydantic import BaseModel, Field 
import instructor 
from enum import Enum
import matplotlib.pyplot as plt 
import seaborn as sns

load_dotenv()

# connect to openai
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError('Oops....missing API key in .env file')

client = instructor.from_openai(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL")))

# sentiment based on feedback email
class SentimentStatus(str, Enum):
    Positive = "positive"
    Negative = "negative"
    Neutral = "neutral"

#category of email
class EmailCategory(str, Enum):
    SUPPORT = "support"
    BILLING = "billing"
    ACCOUNT = "account"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SHIPPING = "shipping"
    CALENDAR = "calendar"
    FEEDBACK = "feedback"
    OTHER = "other"

# email level emergency
class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# define fields of email analysis 
class EmailAnalysis(BaseModel):
    sentiment: SentimentStatus = Field(description='Determine if the email sentiment is positive, negative, or neutral.')
    sentiment_score: int = Field(ge=1, le=10,
                      description='Estimate the sentiment score from 1 (very negative) to 10 (very positive).')
    category: EmailCategory = Field(description='The primary category of the email based on content.')
    urgency: UrgencyLevel = Field(description='Assess the urgency level of the email.')
    key_points: str = Field(description='Brief summary of the main points in the email (1 sentence).')
    action_required: bool = Field(description='Does this email require any action from support team?')

# Your email dataset
csv_data = 'product_category.csv'

df = pd.read_csv(csv_data)
 
print("Original Dataset:")
print(df.head())
print(f"Total emails: {len(df)}")
print("-" * 50)

def analyze_email(text: str) -> EmailAnalysis:
    return client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        response_model=EmailAnalysis,
        messages=[
            {'role': 'system', 'content': 'You are a customer service analytics expert. Analyze emails to extract sentiment, category, urgency, and key points.'},
            {'role': 'user', 'content': text}
        ]
    )

# start chatbot
print('AI initializing...')

results = []

for index, row in df.iterrows():
    analysis = analyze_email(row['interaction_text'])
    
    results.append({
        "interaction_id": row['interaction_id'],
        "email_text": row['interaction_text'],
        "sentiment": analysis.sentiment.value,
        "sentiment_score": analysis.sentiment_score,
        "category_predicted": analysis.category.value,
        "category_original": row['category'],
        "urgency": analysis.urgency.value,
        "key_points": analysis.key_points,
        "action_required": analysis.action_required,
        "last_contact_days": row['last_contact_days'],
        "product_category": row['product_category']
    })
    
    if (index + 1) % 5 == 0:
        print(f"Processed {index + 1}/{len(df)} emails...")

analyzed_df = pd.DataFrame(results)
print("\nAnalysis Complete!")
print(analyzed_df.head())
print("-" * 50)

# Calculate Sentiment Score metrics
def calculate_sentiment_metrics(dataframe):
    total = len(dataframe)
    positive = len(dataframe[dataframe['sentiment'] == 'positive'])
    negative = len(dataframe[dataframe['sentiment'] == 'negative'])
    neutral = len(dataframe[dataframe['sentiment'] == 'neutral'])
    
    positive_pct = (positive/total) * 100
    negative_pct = (negative/total) * 100
    neutral_pct = (neutral/total) * 100
    
    avg_sentiment_score = dataframe['sentiment_score'].mean()
    
    return {
        'total': total,
        'positive': positive,
        'positive_pct': round(positive_pct, 2),
        'negative': negative,
        'negative_pct': round(negative_pct, 2),
        'neutral': neutral,
        'neutral_pct': round(neutral_pct, 2),
        'avg_sentiment_score': round(avg_sentiment_score, 2)
    }

metrics = calculate_sentiment_metrics(analyzed_df)
print("Sentiment Analysis Metrics:")
print(f"Total Emails: {metrics['total']}")
print(f"Positive: {metrics['positive']} ({metrics['positive_pct']}%)")
print(f"Negative: {metrics['negative']} ({metrics['negative_pct']}%)")
print(f"Neutral: {metrics['neutral']} ({metrics['neutral_pct']}%)")
print(f"Average Sentiment Score: {metrics['avg_sentiment_score']}/10")
print("-" * 50)

# Category Accuracy
correct_categories = len(analyzed_df[analyzed_df['category_predicted'] == analyzed_df['category_original']])
accuracy = (correct_categories / len(analyzed_df)) * 100
print(f"Category Prediction Accuracy: {accuracy:.2f}%")
print("-" * 50)

# Urgency Distribution
print("Urgency Distribution:")
print(analyzed_df['urgency'].value_counts())
print("-" * 50)

# Action Required
action_needed = len(analyzed_df[analyzed_df['action_required'] == True])
print(f"Emails requiring action: {action_needed}/{len(analyzed_df)} ({action_needed/len(analyzed_df)*100:.1f}%)")
print("-" * 50)

# Visualization
plt.style.use('ggplot')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

# 1. Sentiment Distribution
sentiment_counts = analyzed_df['sentiment'].value_counts()
colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
bar_colors = [colors.get(sent, '#95a5a6') for sent in sentiment_counts.index]
axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
axes[0, 0].set_title('Email Sentiment Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Number of Emails')
for i, v in enumerate(sentiment_counts.values):
    axes[0, 0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# 2. Sentiment Score Distribution
axes[0, 1].hist(analyzed_df['sentiment_score'], bins=10, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Sentiment Score Distribution (1-10)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(metrics['avg_sentiment_score'], color='red', linestyle='--', 
                   label=f'Avg: {metrics["avg_sentiment_score"]}')
axes[0, 1].legend()

# 3. Category Distribution
category_counts = analyzed_df['category_predicted'].value_counts()
axes[1, 0].barh(category_counts.index, category_counts.values, color='#9b59b6')
axes[1, 0].set_title('Email Categories', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Emails')
for i, v in enumerate(category_counts.values):
    axes[1, 0].text(v + 0.1, i, str(v), va='center', fontweight='bold')

# 4. Urgency by Sentiment
urgency_sentiment = pd.crosstab(analyzed_df['urgency'], analyzed_df['sentiment'])
urgency_sentiment.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[1, 1].set_title('Urgency Level by Sentiment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Urgency Level')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Sentiment')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Save analyzed data to CSV
analyzed_df.to_csv('analyzed_emails.csv', index=False)
print("Analyzed data saved to 'analyzed_emails.csv'")