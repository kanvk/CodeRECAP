from github import Github, GithubException
import pandas as pd
from datetime import datetime, timedelta
import pytz
import concurrent.futures
import time
import random

import os
from github import Github

# Get the GitHub token from environment variables
token = os.getenv("GITHUB_TOKEN")

# Check if the token is retrieved successfully
if token is None:
    raise ValueError("Please set the environment variable 'GITHUB_TOKEN'.")

g = Github(token)



# Initialize a list to store data and set a record limit
data = []
record_limit = 20
repos_limit = 10  # Limit to 10 repositories
prs_per_repo = 10  # Limit to 10 PRs per repository

# Exponential backoff function with jitter
def exponential_backoff(base=60, factor=2, jitter=5):
    return base * factor + random.uniform(-jitter, jitter)

# Process each repository
# Define timeframe for filtering PRs
one_month_ago = datetime.now(pytz.UTC) - timedelta(days=30)  # Keep this offset-aware

# In process_repo function, ensure pr.created_at is also in UTC
def process_repo(repo):
    repo_data = []
    pulls = repo.get_pulls(state='closed')[:prs_per_repo]  # Limit to last 10 PRs

    for pr in pulls:
        try:
            # Convert pr.created_at to UTC timezone to ensure both are comparable
            created_at_utc = pr.created_at.astimezone(pytz.UTC)

            # Check if the PR is merged and meets criteria
            if pr.merged and created_at_utc >= one_month_ago and pr.body and len(pr.body.split()) >= 50:
                instance_id = f"{repo.name}__{pr.number}"

                # Collect base commit details
                base_commit_sha = pr.base.sha
                base_commit = repo.get_commit(base_commit_sha)

                # Initialize commit-related fields
                patch, test_patch = [], []

                # Collect diff/patch information
                for file in pr.get_files():
                    if file.patch:
                        patch.append(file.patch)
                        if "test" in file.filename.lower():
                            test_patch.append(file.patch)

                # Append data for this PR
                repo_data.append({
                    'repo': repo.full_name,
                    'instance_id': instance_id,
                    'base_commit': base_commit_sha,
                    'patch': "\n".join(patch),
                    'test_patch': "\n".join(test_patch),
                    'problem_statement': pr.body,
                    'hints_text': None,  # Placeholder for hints text
                    'created_at': created_at_utc.isoformat(),  # Use ISO format for datetime
                    'version': "v1.0",
                    'FAIL_TO_PASS': None,  # Placeholder for transition metrics
                    'PASS_TO_PASS': None,  # Placeholder for transition metrics
                    'environment_setup_commit': base_commit_sha  # Assuming base commit is setup
                })

                if len(repo_data) >= record_limit:
                    break

        except GithubException as e:
            if e.status == 403:
                # Rate limit hit, apply exponential backoff
                backoff = exponential_backoff()
                print(f"Rate limit hit. Retrying after {backoff:.2f} seconds...")
                time.sleep(backoff)
            else:
                print(f"Error processing PR {pr.number}: {e}")
    
    return repo_data

# Execute with threading and limit to 10 repositories
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_repo, repo) for repo in g.search_repositories('language:Python', sort='stars', order='desc')[:repos_limit]]
    for future in concurrent.futures.as_completed(futures):
        repo_data = future.result()
        data.extend(repo_data)
        if len(data) >= record_limit:
            break

# Convert list to DataFrame
df = pd.DataFrame(data[:record_limit])

# Show the DataFrame
print(df.head())
print(f"Number of records: {len(df)}")

# Optionally, save to CSV
df.to_csv('swe_bench_like_dataset.csv', index=False)
