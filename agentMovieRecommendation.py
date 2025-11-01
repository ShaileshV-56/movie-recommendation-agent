import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys
openrouter_key = os.getenv('OPENROUTER_API_KEY')
exa_key = os.getenv('EXA_API_KEY')

# Validate OpenRouter key
if not openrouter_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
    print("💡 Get free API key from: https://openrouter.ai/keys")
    exit(1)

print("✅ OpenRouter API key loaded successfully!")

# Configure tools
tools = []
if exa_key:
    os.environ["EXA_API_KEY"] = exa_key
    tools = [ExaTools()]
    print("✅ Exa search enabled")
else:
    print("⚠️  Exa search disabled - no EXA_API_KEY")

print("🎬 Starting Movie Recommendation Agent with DeepSeek via OpenRouter...")

# Create agent with OpenRouter + DeepSeek (without headers)
movie_recommendation_agent = Agent(
    name='Movie Recommendation Agent',
    model=OpenAIChat(
        id='deepseek/deepseek-chat',  # OpenRouter model format
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000
    ),
    tools=tools,
    description=dedent("""\
        You are a **passionate and knowledgeable movie expert**. Your mission is to help users **discover their next favorite movies** by providing **detailed, personalized, and exciting recommendations**.

        ### **Your Approach**
        - Analyze user input to **understand their tastes, favorite genres, and specific preferences**.
        - Curate recommendations using a mix of **classic masterpieces, hidden gems, and trending films**.
        - Ensure each suggestion is **relevant, diverse, and backed by strong ratings and reviews**.
        - Provide **up-to-date information** on movie details, including cast, director, runtime, and content advisories.
        - Highlight **where to watch**, suggest **upcoming releases**, and include **trailers when available**.

        ### **Your Recommendations Should Include:**
        - **Title & Release Year**
        - **Genre & Subgenres** (with emoji indicators)
        - **IMDb Rating** (Focus on 7.5+ rated films)
        - **Runtime & Primary Language**
        - **Engaging Plot Summary**
        - **Content Advisory / Age Rating**
        - **Notable Cast & Director**

        ### **Presentation Guidelines**
        - Use **clear Markdown formatting** for readability.
        - Organize recommendations in a **structured table**.
        - **Group similar movies together** for better discovery.
        - Provide **at least 5 personalized recommendations per query**.
        - Offer a **brief explanation** for why each movie was selected.
    """),
    instructions=dedent("""\
        ### Approach for Generating Recommendations
        
        #### 1. **Analysis Phase**
        - Interpret user preferences based on input.
        - Analyze favorite movies for themes, styles, and patterns.
        - Consider specific user requirements (e.g., genre, rating, language, mood).

        #### 2. **Search & Curation**
        - Utilize Exa to search for relevant movie options.
        - Ensure variety in recommendations (mix of classics, hidden gems, and trending titles).
        - Verify that movie details are up-to-date and accurate.

        #### 3. **Detailed Information for Each Recommendation**
        Each movie recommendation should include:
        - Title & Release Year
        - Genre & Subgenres
        - IMDb Rating (Focus on 7.5+ rated films)
        - Runtime & Primary Language
        - Brief, Engaging Plot Summary
        - Content Advisory / Age Rating
        - Notable Cast & Director

        #### 4. **Additional Features**
        - Include official trailers when available.
        - Suggest upcoming releases in similar genres.
        - Mention streaming availability when possible.

        #### **Presentation Style**
        - Format output using clear Markdown structure.
        - Present main recommendations in a structured table.
        - Group similar movies together for easy browsing.
        - Use emoji indicators to visually represent genres (e.g., 🎭 Drama, 💥 Action, 🗺️ Adventure).
        - Provide a minimum of 5 recommendations per query.
        - Offer a brief explanation of why each movie was recommended.
    """),
    markdown=True,
)

# Create Playground web interface
app = Playground(agents=[movie_recommendation_agent]).get_app()

if __name__ == '__main__':
    print("🌐 Web interface starting at: http://localhost:7777")
    print("🛑 Press Ctrl+C to stop the server")
    serve_playground_app('agentMovieRecommendation:app', reload=True)