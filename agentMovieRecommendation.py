import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app, PlaygroundSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment
openrouter_key = os.getenv('OPENROUTER_API_KEY')

# Validate OpenRouter key
if not openrouter_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found")
    print("💡 For Local: Add to .env file")
    print("💡 For Deployment: Add in platform environment variables")
    exit(1)

print("✅ OpenRouter API key loaded successfully!")
print("🎬 Starting Movie Recommendation Agent with DeepSeek via OpenRouter...")

# Create agent with OpenRouter + DeepSeek (without Exa tools)
movie_recommendation_agent = Agent(
    name='Movie Recommendation Agent',
    model=OpenAIChat(
        id='deepseek/deepseek-chat',
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000
    ),
    tools=[],  # No Exa tools - removes the dependency
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
        - Use your extensive movie knowledge to find relevant options.
        - Ensure variety in recommendations (mix of classics, hidden gems, and trending titles).
        - Verify that movie details are accurate based on your training data.

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
        - Include streaming availability when possible.
        - Suggest similar movies in related genres.
        - Mention why each movie suits the user's request.

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

# Create Playground with explicit settings to avoid environment conflicts
playground_settings = PlaygroundSettings(
    # Explicitly set environment to avoid conflicts
    env=None  # or use a valid environment if needed
)

# Create Playground web interface with explicit settings
app = Playground(
    agents=[movie_recommendation_agent],
    settings=playground_settings
).get_app()

if __name__ == '__main__':
    # Get port from environment (for deployment) or default to 7777
    port = int(os.getenv('PORT', 7777))
    
    print(f"🌐 Web interface starting on port: {port}")
    print("🛑 Press Ctrl+C to stop the server")
    
    # For deployment, use 0.0.0.0 to allow external connections
    serve_playground_app(
        'agentMovieRecommendation:app', 
        reload=False,  # Disable reload in production
        host='0.0.0.0',
        port=port
    )