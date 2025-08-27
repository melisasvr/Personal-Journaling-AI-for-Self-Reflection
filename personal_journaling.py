import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import nltk
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

class JournalingAI:
    def __init__(self, db_path="journal.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the SQLite database for storing journal entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            entry TEXT NOT NULL,
            sentiment_score REAL,
            sentiment_label TEXT,
            mood_scale INTEGER,
            topics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_entry(self, entry_text, mood_scale=None, date=None):
        """Add a new journal entry with automatic sentiment analysis and optional mood scale"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Get mood scale if not provided
        if mood_scale is None:
            while True:
                try:
                    mood_input = input("📊 Rate your mood (1-10, where 1=very low, 10=excellent): ").strip()
                    mood_scale = int(mood_input)
                    if 1 <= mood_scale <= 10:
                        break
                    else:
                        print("Please enter a number between 1 and 10.")
                except ValueError:
                    print("Please enter a valid number between 1 and 10.")
        
        # Perform sentiment analysis
        blob = TextBlob(entry_text)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO journal_entries (date, entry, sentiment_score, sentiment_label, mood_scale)
        VALUES (?, ?, ?, ?, ?)
        ''', (date, entry_text, sentiment_score, sentiment_label, mood_scale))
        
        conn.commit()
        conn.close()
        
        print(f"Journal entry added for {date} with {sentiment_label.lower()} sentiment (mood: {mood_scale}/10)")
        return sentiment_score, sentiment_label, mood_scale
    
    def get_entries_df(self):
        """Retrieve all journal entries as a pandas DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM journal_entries ORDER BY date", conn)
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def analyze_sentiment_trends(self, days=30):
        """Analyze sentiment trends over time"""
        df = self.get_entries_df()
        
        if df.empty:
            print("No journal entries found.")
            return None
        
        # Filter to recent entries
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_df = df[df['date'] >= cutoff_date].copy()
        
        if recent_df.empty:
            print(f"No entries found in the last {days} days.")
            return None
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Sentiment over time
        plt.subplot(2, 2, 1)
        plt.plot(recent_df['date'], recent_df['sentiment_score'], marker='o', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Sentiment Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        
        # Plot 2: Sentiment distribution
        plt.subplot(2, 2, 2)
        sentiment_counts = recent_df['sentiment_label'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        
        # Plot 3: Rolling average
        plt.subplot(2, 2, 3)
        if len(recent_df) >= 7:
            recent_df['rolling_sentiment'] = recent_df['sentiment_score'].rolling(window=7).mean()
            plt.plot(recent_df['date'], recent_df['rolling_sentiment'], marker='o', color='red')
            plt.title('7-Day Rolling Average Sentiment')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment')
            plt.xticks(rotation=45)
        
        # Plot 4: Mood scale vs sentiment
        plt.subplot(2, 2, 4)
        if 'mood_scale' in recent_df.columns and recent_df['mood_scale'].notna().any():
            plt.scatter(recent_df['mood_scale'], recent_df['sentiment_score'], alpha=0.6, c=recent_df['mood_scale'], cmap='RdYlGn')
            plt.colorbar(label='Mood Scale')
            plt.title('Your Mood Rating vs Text Sentiment')
            plt.xlabel('Mood Scale (1-10)')
            plt.ylabel('Text Sentiment Score')
        else:
            recent_df['word_count'] = recent_df['entry'].str.split().str.len()
            plt.scatter(recent_df['word_count'], recent_df['sentiment_score'], alpha=0.6)
            plt.title('Entry Length vs Sentiment')
            plt.xlabel('Word Count')
            plt.ylabel('Sentiment Score')
        
        plt.tight_layout()
        plt.show()
        
        return recent_df
    
    def extract_topics(self, n_topics=5, max_features=100):
        """Extract main topics from journal entries using LDA"""
        df = self.get_entries_df()
        
        if df.empty or len(df) < 3:
            print("Need at least 3 entries for topic modeling.")
            return None
        
        # Preprocess text
        def preprocess_text(text):
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            return text
        
        processed_entries = df['entry'].apply(preprocess_text)
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(processed_entries)
        
        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda.fit(tfidf_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        print(f"\n=== Top {n_topics} Topics in Your Journal ===")
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append(top_words)
            print(f"\nTopic {topic_idx + 1}: {', '.join(top_words[:5])}")
        
        return topics, lda, vectorizer
    
    def generate_insights(self):
        """Generate personalized insights based on journal analysis"""
        df = self.get_entries_df()
        
        if df.empty:
            return "No journal entries to analyze yet. Start writing!"
        
        insights = []
        
        # Overall sentiment analysis
        avg_sentiment = df['sentiment_score'].mean()
        if avg_sentiment > 0.1:
            insights.append(f"🌟 Your overall writing tone has been positive (avg: {avg_sentiment:.2f})")
        elif avg_sentiment < -0.1:
            insights.append(f"💙 Your recent entries show some challenges (avg: {avg_sentiment:.2f}). Remember that difficult feelings are valid and temporary.")
        else:
            insights.append(f"⚖️ Your writing shows balanced emotional expression (avg: {avg_sentiment:.2f})")
        
        # Consistency analysis
        days_with_entries = len(df['date'].dt.date.unique())
        total_entries = len(df)
        
        if total_entries >= 7:
            insights.append(f"📝 You've been journaling consistently with {total_entries} entries over {days_with_entries} days")
        
        # Recent trends
        if len(df) >= 5:
            recent_5 = df.tail(5)['sentiment_score'].mean()
            earlier_5 = df.head(5)['sentiment_score'].mean() if len(df) >= 10 else df.iloc[:-5]['sentiment_score'].mean()
            
            if recent_5 > earlier_5 + 0.1:
                insights.append("📈 Your recent entries show improving mood compared to earlier ones")
            elif recent_5 < earlier_5 - 0.1:
                insights.append("📉 Recent entries show some emotional challenges. Consider what support might help")
        
        # Entry length insights
        avg_length = df['entry'].str.len().mean()
        if avg_length > 500:
            insights.append("📚 You tend to write detailed entries - great for processing complex thoughts!")
        elif avg_length < 150:
            insights.append("✏️ You prefer concise entries - sometimes expanding on your thoughts can reveal new insights")
        
        # Mood scale insights
        if 'mood_scale' in df.columns and df['mood_scale'].notna().any():
            avg_mood = df['mood_scale'].mean()
            insights.append(f"🎯 Your average mood rating is {avg_mood:.1f}/10")
            
            # Mood trend
            if len(df) >= 5:
                recent_mood = df.tail(5)['mood_scale'].mean()
                earlier_mood = df.head(5)['mood_scale'].mean() if len(df) >= 10 else df.iloc[:-5]['mood_scale'].mean()
                
                if recent_mood > earlier_mood + 0.5:
                    insights.append("📈 Your mood ratings have been improving recently")
                elif recent_mood < earlier_mood - 0.5:
                    insights.append("📉 Your mood ratings show recent challenges - consider self-care strategies")
        
        return "\n".join(insights)
    
    def suggest_reflection_questions(self):
        """Generate personalized reflection questions based on recent entries"""
        df = self.get_entries_df()
        
        if df.empty:
            return ["What's one thing you're grateful for today?",
                   "How are you feeling right now, and why?",
                   "What's been on your mind lately?"]
        
        recent_sentiment = df.tail(3)['sentiment_score'].mean() if len(df) >= 3 else df['sentiment_score'].mean()
        
        questions = []
        
        if recent_sentiment > 0.2:
            questions.extend([
                "What specific moments brought you joy recently?",
                "How can you build on the positive patterns you're noticing?",
                "What strengths have you discovered about yourself lately?"
            ])
        elif recent_sentiment < -0.2:
            questions.extend([
                "What's one small step you could take to care for yourself today?",
                "Who in your life makes you feel supported?",
                "What would you tell a friend going through something similar?"
            ])
        else:
            questions.extend([
                "What's something you've learned about yourself this week?",
                "How have your priorities or values evolved recently?",
                "What are you most curious about in your life right now?"
            ])
        
        # Add some general reflection questions
        questions.extend([
            "What patterns do you notice in your thoughts and emotions?",
            "How do you want to grow in the coming weeks?",
            "What would make tomorrow feel meaningful to you?"
        ])
        
        return np.random.choice(questions, size=3, replace=False).tolist()
    
    def generate_weekly_report(self, weeks_back=1):
        """Generate a comprehensive weekly report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks_back)
        
        df = self.get_entries_df()
        if df.empty:
            return "No entries available for weekly report."
        
        # Filter to the specified week
        week_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        
        if week_df.empty:
            return f"No entries found for the past {weeks_back} week(s)."
        
        report = []
        report.append("📅 WEEKLY JOURNAL REPORT")
        report.append("=" * 40)
        report.append(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        report.append(f"Total entries: {len(week_df)}")
        
        # Sentiment analysis
        avg_sentiment = week_df['sentiment_score'].mean()
        sentiment_counts = week_df['sentiment_label'].value_counts()
        
        report.append(f"\n📊 EMOTIONAL OVERVIEW:")
        report.append(f"   Average sentiment: {avg_sentiment:.2f}")
        report.append(f"   Positive entries: {sentiment_counts.get('Positive', 0)}")
        report.append(f"   Negative entries: {sentiment_counts.get('Negative', 0)}")
        report.append(f"   Neutral entries: {sentiment_counts.get('Neutral', 0)}")
        
        # Mood scale analysis
        if 'mood_scale' in week_df.columns and week_df['mood_scale'].notna().any():
            avg_mood = week_df['mood_scale'].mean()
            min_mood = week_df['mood_scale'].min()
            max_mood = week_df['mood_scale'].max()
            
            report.append(f"\n🎯 MOOD TRACKING:")
            report.append(f"   Average mood: {avg_mood:.1f}/10")
            report.append(f"   Mood range: {min_mood} - {max_mood}")
            report.append(f"   Days above 7/10: {len(week_df[week_df['mood_scale'] >= 7])}")
            report.append(f"   Days below 4/10: {len(week_df[week_df['mood_scale'] <= 3])}")
        
        # Most common themes
        all_text = ' '.join(week_df['entry'].tolist())
        common_words = []
        try:
            # Simple word frequency (avoiding complex NLP for quick insights)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
            stop_words = {'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'about', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'new', 'years', 'way', 'may', 'days', 'much', 'these', 'people', 'all', 'any', 'many', 'now', 'get', 'most', 'made', 'after', 'back', 'other', 'many', 'well', 'large', 'must', 'still', 'should', 'being'}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            word_freq = Counter(filtered_words)
            common_words = word_freq.most_common(5)
        except:
            common_words = [("analysis", 1), ("unavailable", 1)]
        
        if common_words:
            report.append(f"\n🏷️ COMMON THEMES:")
            for word, count in common_words:
                report.append(f"   • {word.title()} (mentioned {count} times)")
        
        # Recommendations
        report.append(f"\n💡 INSIGHTS & RECOMMENDATIONS:")
        if avg_sentiment > 0.2:
            report.append("   • You've had a predominantly positive week! Keep up the good energy.")
        elif avg_sentiment < -0.1:
            report.append("   • This week showed some challenges. Consider self-care activities.")
        else:
            report.append("   • You've had a balanced emotional week with mixed experiences.")
        
        if 'mood_scale' in week_df.columns and week_df['mood_scale'].notna().any():
            if avg_mood >= 7:
                report.append("   • Your mood ratings indicate a great week overall!")
            elif avg_mood <= 4:
                report.append("   • Your mood ratings suggest some difficult days. Be gentle with yourself.")
        
        if len(week_df) >= 5:
            report.append("   • Excellent journaling consistency this week!")
        elif len(week_df) >= 3:
            report.append("   • Good journaling habit - try to write a bit more regularly.")
        else:
            report.append("   • Consider journaling more frequently for better self-awareness.")
        
        return "\n".join(report)
    
    def export_analysis(self, filename=None):
        """Export comprehensive analysis to a text file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"journal_analysis_{timestamp}.txt"
        
        df = self.get_entries_df()
        if df.empty:
            print("No data to export.")
            return None
        
        # Generate comprehensive analysis
        analysis_parts = []
        
        # Header
        analysis_parts.append("PERSONAL JOURNAL ANALYSIS REPORT")
        analysis_parts.append("=" * 50)
        analysis_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis_parts.append(f"Total Entries: {len(df)}")
        analysis_parts.append("")
        
        # Overall insights
        analysis_parts.append("📊 OVERALL INSIGHTS:")
        analysis_parts.append("-" * 20)
        insights = self.generate_insights()
        analysis_parts.append(insights)
        analysis_parts.append("")
        
        # Weekly report
        analysis_parts.append("📅 RECENT WEEKLY REPORT:")
        analysis_parts.append("-" * 25)
        weekly_report = self.generate_weekly_report()
        analysis_parts.append(weekly_report)
        analysis_parts.append("")
        
        # Topics
        analysis_parts.append("🏷️ MAIN TOPICS:")
        analysis_parts.append("-" * 15)
        try:
            topics, _, _ = self.extract_topics(n_topics=5)
            for i, topic in enumerate(topics, 1):
                analysis_parts.append(f"Topic {i}: {', '.join(topic[:8])}")
        except:
            analysis_parts.append("Topic analysis unavailable (need more entries)")
        analysis_parts.append("")
        
        # Reflection questions
        analysis_parts.append("❓ REFLECTION QUESTIONS:")
        analysis_parts.append("-" * 25)
        questions = self.suggest_reflection_questions()
        for i, question in enumerate(questions, 1):
            analysis_parts.append(f"{i}. {question}")
        analysis_parts.append("")
        
        # Entry summary
        analysis_parts.append("📝 ENTRY SUMMARY:")
        analysis_parts.append("-" * 17)
        sentiment_counts = df['sentiment_label'].value_counts()
        analysis_parts.append(f"Positive entries: {sentiment_counts.get('Positive', 0)}")
        analysis_parts.append(f"Negative entries: {sentiment_counts.get('Negative', 0)}")
        analysis_parts.append(f"Neutral entries: {sentiment_counts.get('Neutral', 0)}")
        
        if 'mood_scale' in df.columns and df['mood_scale'].notna().any():
            avg_mood = df['mood_scale'].mean()
            analysis_parts.append(f"Average mood rating: {avg_mood:.1f}/10")
        
        # Write to file
        full_analysis = "\n".join(analysis_parts)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_analysis)
            print(f"✅ Analysis exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error exporting file: {e}")
            return None

# Example usage and demo
def demo_journaling_ai():
    """Demonstrate the journaling AI with sample entries"""
    ai = JournalingAI()
    
    # Sample entries for demonstration (using current dates) - added silently
    current_date = datetime.now()
    sample_entries = [
        ((current_date - timedelta(days=20)).strftime("%Y-%m-%d"), "Started feeling hopeful and excited about new possibilities. Set some meaningful goals for personal growth."),
        ((current_date - timedelta(days=18)).strftime("%Y-%m-%d"), "Had a challenging day at work. Feeling overwhelmed with the new project deadlines and responsibilities."),
        ((current_date - timedelta(days=15)).strftime("%Y-%m-%d"), "Spent quality time with family today. Really grateful for their support and love. Feeling much better."),
        ((current_date - timedelta(days=12)).strftime("%Y-%m-%d"), "Struggled with anxiety today. Hard to focus on tasks. Maybe I need to practice more self-care."),
        ((current_date - timedelta(days=8)).strftime("%Y-%m-%d"), "Amazing breakthrough in my creative project! Feeling inspired and proud of the progress I've made."),
        ((current_date - timedelta(days=5)).strftime("%Y-%m-%d"), "Reflecting on friendship and relationships. Realized how important it is to invest time in people who matter."),
        ((current_date - timedelta(days=2)).strftime("%Y-%m-%d"), "Feeling stuck and unmotivated. Need to find ways to reignite my passion for my goals.")
    ]
    
    print("=== Personal Journaling AI ===")
    print("Starting with some sample data in the background...")
    
    # Add sample entries silently (no print statements)
    for date, entry in sample_entries:
        # Perform sentiment analysis
        blob = TextBlob(entry)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Assign mood scale based on sentiment (for demo data)
        if sentiment_score > 0.3:
            mood_scale = np.random.randint(7, 10)
        elif sentiment_score < -0.2:
            mood_scale = np.random.randint(2, 5)
        else:
            mood_scale = np.random.randint(5, 8)
        
        # Store in database silently
        conn = sqlite3.connect(ai.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO journal_entries (date, entry, sentiment_score, sentiment_label, mood_scale)
        VALUES (?, ?, ?, ?, ?)
        ''', (date, entry, sentiment_score, sentiment_label, mood_scale))
        conn.commit()
        conn.close()
    
    return ai

def show_final_analysis(ai):
    """Show comprehensive analysis at the end"""
    print("\n" + "="*60)
    print("📊 FINAL ANALYSIS & SUMMARY")
    print("="*60)
    
    # Get all entries for summary
    df = ai.get_entries_df()
    if df.empty:
        print("No entries to analyze.")
        return
    
    # Show entry summary by sentiment
    sentiment_counts = df['sentiment_label'].value_counts()
    print(f"\n📝 Your Journal Summary:")
    print(f"   Total entries: {len(df)}")
    print(f"   Positive entries: {sentiment_counts.get('Positive', 0)}")
    print(f"   Negative entries: {sentiment_counts.get('Negative', 0)}")
    print(f"   Neutral entries: {sentiment_counts.get('Neutral', 0)}")
    
    # Mood scale summary
    if 'mood_scale' in df.columns and df['mood_scale'].notna().any():
        avg_mood = df['mood_scale'].mean()
        print(f"   Average mood rating: {avg_mood:.1f}/10")
        print(f"   Best day: {df.loc[df['mood_scale'].idxmax(), 'date'].strftime('%Y-%m-%d')} ({df['mood_scale'].max()}/10)")
        print(f"   Challenging day: {df.loc[df['mood_scale'].idxmin(), 'date'].strftime('%Y-%m-%d')} ({df['mood_scale'].min()}/10)")
    
    # Generate insights
    print(f"\n📊 Personal Insights:")
    insights = ai.generate_insights()
    print(insights)
    
    # Extract topics
    print(f"\n🏷️ Main Topics in Your Entries:")
    topics, _, _ = ai.extract_topics(n_topics=3)
    
    # Suggest reflection questions
    print(f"\n❓ Reflection Questions Based on Your Writing:")
    questions = ai.suggest_reflection_questions()
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    # Show sentiment trends with visualization
    print(f"\n📈 Your Emotional Journey - Visualizing Trends:")
    ai.analyze_sentiment_trends(days=30)

if __name__ == "__main__":
    # Initialize with sample data quietly
    journal_ai = demo_journaling_ai()
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE JOURNALING MODE")
    print("="*60)
    print("\n📝 Add your own journal entries!")
    print("Commands: 'quit' to exit | 'analyze' for insights | 'questions' for prompts")
    print("          'weekly' for weekly report | 'export' to save analysis")
    
    while True:
        user_input = input("\n✏️  Enter your journal entry (or command): ").strip()
        
        if user_input.lower() == 'quit':
            # Show final comprehensive analysis when user quits
            show_final_analysis(journal_ai)
            print("\n👋 Thank you for journaling! Keep reflecting and growing.")
            break
        elif user_input.lower() == 'analyze':
            print("\n📊 Current Insights:")
            print(journal_ai.generate_insights())
        elif user_input.lower() == 'questions':
            questions = journal_ai.suggest_reflection_questions()
            print("\n❓ Reflection Questions:")
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q}")
        elif user_input.lower() == 'weekly':
            print("\n📅 Weekly Report:")
            print(journal_ai.generate_weekly_report())
        elif user_input.lower() == 'export':
            print("\n📁 Exporting analysis...")
            filename = journal_ai.export_analysis()
            if filename:
                print(f"Your complete analysis has been saved!")
        elif len(user_input.strip()) > 10:
            try:
                sentiment_score, sentiment_label, mood_scale = journal_ai.add_entry(user_input)
                print(f"✅ Entry saved! Sentiment: {sentiment_label.lower()} ({sentiment_score:.2f}) | Mood: {mood_scale}/10")
            except KeyboardInterrupt:
                print("\n⏭️  Skipping mood rating...")
                # Fallback to add entry without mood interaction
                blob = TextBlob(user_input)
                sentiment_score = blob.sentiment.polarity
                sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
                
                conn = sqlite3.connect(journal_ai.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO journal_entries (date, entry, sentiment_score, sentiment_label, mood_scale)
                VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now().strftime("%Y-%m-%d"), user_input, sentiment_score, sentiment_label, None))
                conn.commit()
                conn.close()
                print(f"✅ Entry saved! Sentiment: {sentiment_label.lower()} ({sentiment_score:.2f})")
        else:
            print("💭 Please write a longer entry (at least 10 words) or use a command.")