import pandas as pd

file_before = pd.read_csv('cleaned_posts_before.csv')
file_after = pd.read_csv('cleaned_posts_after.csv')

file_before['created'] = pd.to_datetime(file_before['created'], unit='s')
file_after['created'] = pd.to_datetime(file_after['created'], unit='s')

before_subreddit_stats = file_before.groupby('subreddit')[['ups', 'num_comments', 'upvote_ratio']].mean()
after_subreddit_stats = file_after.groupby('subreddit')[['ups', 'num_comments', 'upvote_ratio']].mean()

subreddit_diff = after_subreddit_stats - before_subreddit_stats

print("Average values by subreddit before November 10:")
print(before_subreddit_stats)
print("\nAverage values by subreddit after November 10:")
print(after_subreddit_stats)
print("\nDifference in mean values (after - before):")
print(subreddit_diff)
