# Kalshi GenAI Trading Bot
This is a vibe coded Generative AI prediction market trading bot. The goal of this experiment is to create a role prompt for an LLM (in this case Grok) to make it into a professional prediction market trader. Currently, v1 of this project only has a single bot, which takes new positions. Over time, I plan to add another both for re-evaluating and selling existing positions.

## Getting started with the New Position Bot
1. Clone the repo.
2. Copy your Kalshi private key pem to the root directory, and name it "kalshi_private_key".
3. Rename the .env.example file to .env, and add your Kalshi API key and Grok API key. Edit the other ENV variables as desired.
3. Change directory into the New Position Bot directory: `cd new_position_bot`
4. Ensure you're on Python 3.13.11, and run `pip install -r requirements.txt`.
5. In the project root, run `python main.py`.

## New Position Bot design notes
*Vibe coded: As this bot will use an LLM to make decisions surrounding prediction market trades, it seemed appropriate to also have the LLM write the initial code for the project. The many iterations of prompts used to generate this codebase are found in `/prompts`.
*Role prompting: The prompt sent to the LLM tells it that it's a professional prediction market trader, in the hopes of enhancing the output quality. 
*Serverless: The bot is designed to run as two serverless functions on Google Cloud Platform, one to place new trades and one to exit markets. By default it runs hourly, and looks at all existing markets in which it has not yet taken a position. If it finds a market in which it has not taken a position, it will ask the LLM if it should take a position, and act on that decision. 

### Why Grok? 
*X.com data access: With recent Terms of Service updates, X has restricted which models can access it's data to just Grok. As prediction market trades are very much based on real-time data, I believe this access is critical for deciding which markets to enter and exit.
*Extensibility: The xAI API is designed to be interchangeable with ChatGPT and Gemini, making it easy to swap Grok out in the future. 

## New Position Bot file breakdown 
1. Main.py: Entry point that calls the Kalshi and Grok clients.
2. Kalshi_client.py: Client for making API calls to Kalshi.
3. Grok_client.py: Client for making API calls to xAI's Grok.
4. Utils.py: Loads ENV varibales and the Kalshi private key pem.

## Contributing
1. Fork this repo.
2. Clone your fork to your local machine.
3. Cut a branch of main named `feature/cool-feature-name` or `bugfix/bug-to-be-fixed`.
4. Make your changes and run the tests with `python -m pytest` from the `new_position_bot/` directory.
5. Assuming the tests pass, commit your local changes.
5. Push your changes up to your forked repo.
6. Open a PR from your forked repo's feature or bugfix branch to this repo's main branch.
