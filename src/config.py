# src/config.py
import os

# Prefer environment variables (great for Docker),
# but fall back to hardcoded values if not set.
BEARER_TOKEN = os.getenv("AAAAAAAAAAAAAAAAAAAAADhm3gEAAAAALOH%2FqKz4OrZCMIsck%2FGg%2B7YgVmk%3DE43rJv7OvnRjFWq9lW7GsnYABlznjNUH4h7V1DTqB6ga8DIP97")
HASHTAG = os.getenv("HASHTAG", "#YourCampaignHashtag")

