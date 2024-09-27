# import os

# # Load environment variables
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# # If the environment variable is not set, use a default value or raise an error
# if OPENAI_API_KEY is None:
#     # Option 1: Use a default value (not recommended for production)
#     # OPENAI_API_KEY = 'your-api-key-here'
    
#     # Option 2: Raise an error
#     raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

# # You can add more environment variables here as needed
# # For example:
# # DATABASE_URL = os.environ.get('DATABASE_URL')
# # DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# # Function to check if all required environment variables are set
# def check_env_variables():
#     required_vars = ['OPENAI_API_KEY']
#     missing_vars = [var for var in required_vars if not globals().get(var)]
#     if missing_vars:
#         raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# # Run the check when the module is imported
# check_env_variables()

OPENAI_API_KEY = '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'