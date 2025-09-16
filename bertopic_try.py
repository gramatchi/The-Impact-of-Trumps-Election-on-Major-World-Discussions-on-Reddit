import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

def load_data(file_path):
    """Loads a CSV file and returns a DataFrame"""
    df = pd.read_csv(file_path)
    return df

def extract_topics(df, column="text"):
    """Extracts topics from text using BERTopic"""
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(df[column].astype(str).tolist())
    return topic_model, topics

def compare_topics(topic_model_before, topic_model_after, size_before, size_after):
    """Compares topic popularity before and after a certain date, considering dataset size"""
    freq_before = topic_model_before.get_topic_freq()
    freq_after = topic_model_after.get_topic_freq()
    
    freq_before.columns = ["Topic", "Count_Before"]
    freq_after.columns = ["Topic", "Count_After"]
    
    comparison = pd.merge(freq_before, freq_after, on="Topic", how="outer").fillna(0)
    
    # Normalize by dataset size
    comparison["Proportion_Before"] = comparison["Count_Before"] / size_before
    comparison["Proportion_After"] = comparison["Count_After"] / size_after
    comparison["Change_Proportion"] = (comparison["Proportion_After"] - comparison["Proportion_Before"]) * 100  # In percentage
    
    # Add topic names instead of numbers
    def get_topic_name(topic_model, topic_id):
        topic_info = topic_model.get_topic(topic_id)
        return ", ".join([word for word, _ in topic_info]) if topic_info else "Unknown"
    
    comparison["Topic_Name"] = comparison["Topic"].apply(lambda x: get_topic_name(topic_model_before, x) if x != -1 else "Unknown")
    comparison = comparison[["Topic_Name", "Proportion_Before", "Proportion_After", "Change_Proportion"]]
    
    return comparison

def visualize_topic_changes(comparison_df):
    """Visualizes changes in topic popularity"""
    top_growing = comparison_df.sort_values(by="Change_Proportion", ascending=False).head(10)
    top_falling = comparison_df.sort_values(by="Change_Proportion", ascending=True).head(10)
    
    plt.figure(figsize=(16, 10))
    plt.barh(top_growing["Topic_Name"], top_growing["Change_Proportion"], color="green")
    plt.xlabel("Change in Popularity (%)")
    plt.ylabel("Topics")
    plt.title("Top 10 Growing Topics")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.gca().invert_yaxis()
    plt.show()
    
    plt.figure(figsize=(16, 10))
    plt.barh(top_falling["Topic_Name"], top_falling["Change_Proportion"], color="red")
    plt.xlabel("Change in Popularity (%)")
    plt.ylabel("Topics")
    plt.title("Top 10 Declining Topics")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    file_before = "cleaned_posts_before.csv"
    file_after = "cleaned_posts_after.csv"
    
    df_before = load_data(file_before)
    df_after = load_data(file_after)
    
    size_before = len(df_before)
    size_after = len(df_after)
    
    print("Extracting topics...")
    topic_model_before, _ = extract_topics(df_before)
    topic_model_after, _ = extract_topics(df_after)
    
    print("Comparing topic popularity...")
    comparison_df = compare_topics(topic_model_before, topic_model_after, size_before, size_after)
    
    print(comparison_df.sort_values(by="Change_Proportion", ascending=False))
    
    visualize_topic_changes(comparison_df)
