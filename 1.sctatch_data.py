import praw
import pandas as pd
from tqdm import tqdm

reddit = praw.Reddit(
    client_id='kn0BxRmV_yMcqHPeDL9uQA',
    client_secret='szedxiFdgxiIYukDVNXm_-gYnOnNOA',
    user_agent='Dry_Try8800',
    check_for_async=False
)

keywords = ['ElonMusk', 'Biden', 'Trump', 'Republicans', 'Democrats', 'KamalaHarris', 'Europe',
            'Ukraine', 'Russia', 'Israel', 'Palestine', 'ClimateChange']

def get_top_subreddits_for_keywords(keywords):
    keyword_subreddits = {}
    
    for keyword in keywords:
        print(f"\nSearching for subreddits related to the hashtag '{keyword}'...")
        related_subreddits = reddit.subreddits.search_by_name(keyword, include_nsfw=False, exact=False)
        top_subreddits = list(related_subreddits)[:5]
        
        if not top_subreddits:
            print(f"No subreddits found related to '{keyword}'.")
            keyword_subreddits[keyword] = []
        else:
            print("\nTop 5 subreddits related to your keyword:")
            for i, subreddit in enumerate(top_subreddits):
                print(f"{i + 1}. {subreddit.display_name}")
            keyword_subreddits[keyword] = [subreddit.display_name for subreddit in top_subreddits]
    
    return keyword_subreddits

top_subreddits_per_keyword = get_top_subreddits_for_keywords(keywords)

all_posts = []
total_posts_to_retrieve = 1000
time_filter = 'year'

for keyword, subreddits in top_subreddits_per_keyword.items():
    if subreddits:
        subreddit_name = subreddits[0]
        print(f"\nRetrieving {total_posts_to_retrieve} posts from r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        
        for post in tqdm(subreddit.top(limit=total_posts_to_retrieve, time_filter=time_filter),
                         total=total_posts_to_retrieve, desc=f'Reddit posts for {keyword}'):
            all_posts.append({
                'keyword': keyword,
                'subreddit': post.subreddit.display_name,
                'selftext': post.selftext,
                'author_fullname': post.author_fullname if post.author else 'N/A',
                'title': post.title,
                'upvote_ratio': post.upvote_ratio,
                'ups': post.ups,
                'created': post.created,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author) if post.author else 'N/A',
                'id': post.id
            })

df = pd.DataFrame(all_posts)
df.drop_duplicates(subset='id', inplace=True)
df.to_csv('posts.xlsx', index=False)
