"""
Schwab API Token Re-authentication Helper
==========================================

This script guides you through re-authenticating with the Schwab API
to refresh the expired OAuth token.

Author: AI Assistant
Date: February 16, 2026
"""

import json
from datetime import datetime
from schwab.auth import easy_client

# Schwab API credentials
API_KEY = 'BR0o3XFTyp4HUy5z7dWYX3IzgWGlJrIN'
APP_SECRET = 'mgG7N0t2AKGSAIKu'
CALLBACK_URL = 'https://127.0.0.1:8182'
TOKEN_PATH = 'schwab_token.json'


def check_token_status():
    """Check current token status and expiration."""
    try:
        with open(TOKEN_PATH, 'r') as f:
            token_data = json.load(f)
        
        expires_at = token_data.get('token', {}).get('expires_at', 0)
        creation_timestamp = token_data.get('creation_timestamp', 0)
        
        expires_dt = datetime.fromtimestamp(expires_at)
        creation_dt = datetime.fromtimestamp(creation_timestamp)
        now = datetime.now()
        
        is_expired = now > expires_dt
        
        print("\n" + "="*60)
        print("SCHWAB API TOKEN STATUS")
        print("="*60)
        print(f"Token created: {creation_dt}")
        print(f"Token expires: {expires_dt}")
        print(f"Current time:  {now}")
        print(f"Status: {'🔴 EXPIRED' if is_expired else '🟢 VALID'}")
        print(f"Time until expiration: {expires_dt - now if not is_expired else 'Already expired'}")
        print("="*60 + "\n")
        
        return not is_expired
        
    except FileNotFoundError:
        print("No token file found. Will create new token.")
        return False


def authenticate_and_save_token():
    """Authenticate with Schwab and save token."""
    print("\nStarting Schwab OAuth authentication...")
    print("A browser window will open for you to log in.\n")
    
    try:
        client = easy_client(
            api_key=API_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK_URL,
            token_path=TOKEN_PATH,
            enforce_enums=False
        )
        
        print("\n✅ Authentication successful!")
        print(f"✅ Token saved to: {TOKEN_PATH}")
        
        # Check token status
        check_token_status()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you accept the security warning about the SSL certificate")
        print("2. Check that your Schwab account login is correct")
        print("3. Ensure you're connected to the internet")
        print("4. Try again with: python setup_schwab_token.py")
        return False


def main():
    """Main flow for token setup/refresh."""
    print("\n" + "="*60)
    print("SCHWAB API TOKEN SETUP/REFRESH")
    print("="*60)
    
    # Check current token status
    token_valid = check_token_status()
    
    if token_valid:
        print("✅ Token is still valid. No action needed.")
        print("\nYour Internals data fetching should work without re-authentication.")
        response = input("Do you want to refresh the token anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Authenticate and save
    success = authenticate_and_save_token()
    
    if success:
        print("\n✅ Token refresh complete!")
        print("You can now run: python internals_data.py")
    else:
        print("\n❌ Token refresh failed.")
        print("Please check the error messages above and try again.")


if __name__ == '__main__':
    main()
