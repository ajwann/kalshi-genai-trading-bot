This is a vibe coded Generative AI prediction market trading bot. The goal of this toy project is to create a role pormpt for an LLM (in this case Grok) to make it into a professional prediciotn market trader. The design also relies on Retrievval Augemented Generation (RAG), to have the LLM pull in external resources when it's prompted.  

Design notes
*Vibe coded: As this bot will use an LLM to make decisions surrounding prediciotnj market trades, it seemed appropriate to also have the LLM write the initial code for the project. The many iterations of prompts used to generate this code-base are found in ```/prompts```.
**Notes on code generation prompt: In the ```/prompts``` directory you can see how my original code generation prompt evolved vrom v1 to v13. The final version was succesfull due to the incorporation of Retrieval Augmented Generation. I fed the LLM links to the Kalshi API docs, Python 3.13 docs, the xAI API docs, and the Python style guide. 
*Role prompting: The prompt that this bot sends to the LLM tells it that it's a professional prediciton market trader, in the hopes of encouraging the LLM to think as if it were a professional prediction market trader. 
*RAG: When prmpted,m the LLM is provided with external resources to access, such as Wikipedia, Grokipedia, and
*Serverless: The bot is designed to run as two serverless funcitons on Google Cloud Platform, one to place new trades and one to exit markets. By default it runs hourly, and looks at all existing markets in which it has not yet taken a position. If it finds a market in which it has not taken a positioin, it will ask the LLM if it should take a position, and act on that decision. 

Why Grok? 
*Extensibility: The xAI API is designed to be interchangable with ChatGPT and Gemini, making it easy to swap Grok out in the future. 
*X.com data access: With recent Terms of Service updates, X has restricted which models can access it's data to just Grok. As prediction market trades are very much based on real-time data, I belive this access is critical for deciding which markets to enter and exit.
