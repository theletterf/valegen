
import os
import sys
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

# --- Configuration & Initialization ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

DB_BASE_PATH = os.path.join(os.path.dirname(__file__), "vectordb")

def get_db_path_for_provider(provider):
    """Get database path specific to the provider's embedding model."""
    if provider == "gemini":
        return os.path.join(DB_BASE_PATH, "gemini_768d")
    elif provider in ["openai", "claude"]:
        return os.path.join(DB_BASE_PATH, "openai_3072d")
    else:
        return DB_BASE_PATH
# Global variables for dynamic provider switching
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Global model instances (will be reinitialized when provider changes)
current_embeddings = None
current_llm = None
current_vectordb = None
current_base_retriever = None

def get_embedding_model():
    """Initialize embeddings model based on provider."""
    if AI_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    elif AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider") 
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    elif AI_PROVIDER == "claude":
        # Claude doesn't have a dedicated embedding model, use OpenAI as fallback
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for embeddings when using Claude provider")
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

def get_chat_model():
    """Initialize chat model based on provider."""
    if AI_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        return ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True
        )
    elif AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
        return ChatOpenAI(
            model="gpt-5-mini",  # Using GPT-4o as GPT-5 is not yet available
            api_key=OPENAI_API_KEY,
            temperature=0.0
        )
    elif AI_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY required for Claude provider")
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Latest Sonnet model
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
            max_tokens=4096
        )
    else:
        raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

def initialize_models():
    """Initialize all AI models and components."""
    global current_embeddings, current_llm, current_vectordb, current_base_retriever
    
    current_embeddings = get_embedding_model()
    db_path = get_db_path_for_provider(AI_PROVIDER)
    current_vectordb = Chroma(persist_directory=db_path, embedding_function=current_embeddings)
    current_base_retriever = current_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    current_llm = get_chat_model()
    
    return current_embeddings, current_llm, current_vectordb, current_base_retriever

def switch_provider(new_provider):
    """Switch to a new AI provider and reinitialize models."""
    global AI_PROVIDER
    
    if new_provider not in ['gemini', 'openai', 'claude']:
        raise ValueError(f"Unsupported provider: {new_provider}")
    
    # Check if provider is available
    available_providers = [p['id'] for p in get_available_providers()]
    if new_provider not in available_providers:
        raise ValueError(f"Provider {new_provider} is not available (missing API keys)")
    
    AI_PROVIDER = new_provider
    initialize_models()
    print(f"✓ Switched to {AI_PROVIDER.upper()} provider")

def get_available_providers():
    """Get list of available providers based on configured API keys."""
    providers = []
    
    # Check Gemini
    if GEMINI_API_KEY:
        providers.append({
            'id': 'gemini',
            'name': 'Google Gemini',
            'description': 'Latest Gemini model with advanced reasoning'
        })
    
    # Check OpenAI
    if OPENAI_API_KEY:
        providers.append({
            'id': 'openai', 
            'name': 'OpenAI GPT',
            'description': 'OpenAI\'s most capable model'
        })
    
    # Check Claude (requires both OpenAI for embeddings and Anthropic for chat)
    if ANTHROPIC_API_KEY and OPENAI_API_KEY:
        providers.append({
            'id': 'claude',
            'name': 'Claude',
            'description': 'Anthropic\'s latest reasoning model'
        })
    
    return providers

# --- Pre-flight Checks ---
def check_vector_databases():
    """Check if vector databases exist for available providers."""
    available_providers = [p['id'] for p in get_available_providers()]
    missing_dbs = []
    
    for provider in available_providers:
        db_path = get_db_path_for_provider(provider)
        if not os.path.exists(db_path):
            missing_dbs.append(provider)
    
    if missing_dbs:
        print(f"\nFATAL ERROR: Vector database(s) not found for: {', '.join(missing_dbs)}")
        print("Please run 'python setup.py' to build the required databases.")
        sys.exit(1)
    
    print(f"✓ Vector databases found for all available providers")

check_vector_databases()

print(f"Using AI provider: {AI_PROVIDER.upper()}")

# Validate API keys based on provider
try:
    get_embedding_model()  # This will check for required API keys
    get_chat_model()
except ValueError as e:
    print(f"\nFATAL ERROR: {e}")
    print("Please set the required API key in your .env file.")
    sys.exit(1)

# --- LangChain Components Initialization ---
try:
    embeddings, llm, vectordb, base_retriever = initialize_models()
    print(f"✓ AI components loaded successfully using {AI_PROVIDER.upper()}.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not initialize AI components: {e}")
    sys.exit(1)

# --- RAG Chain Definition ---
RULE_PROMPT_TEMPLATE = """
SYSTEM:
You are an expert in writing linter rules for Vale (https://vale.sh).
Your task is to generate THREE different valid Vale rules with confidence scores based on the user's request.
Use the provided context which includes BOTH Vale documentation AND working rule examples to understand the available options and syntax.

IMPORTANT: When you see "EXAMPLE RULE" entries in the context, these are real, working Vale rules. 
Use these examples as templates and patterns for your generated rule. Pay close attention to:
- The exact YAML structure and field names used in examples
- How different rule types implement their specific functionality
- Required vs optional fields for each rule type

Vale Rule Types:

1. existence: Checks for presence/absence of regex patterns
   - Pattern matching with tokens array
   - Use for forbidden words/phrases
   - Required fields: tokens or pattern

2. substitution: Suggests replacements for patterns
   - Uses swap dictionary for find/replace
   - Supports regex patterns and simple strings
   - Required fields: swap

3. occurrence: Ensures patterns appear specific number of times
   - min/max constraints for pattern frequency
   - Required fields: pattern, min/max

4. consistency: Ensures uniform usage across document
   - Checks for consistent terminology/capitalization
   - Required fields: either (for alternatives) or tokens

5. capitalization: Validates capitalization rules
   - match: sentence|title|lower|upper
   - exceptions array for special cases
   - Required fields: match

6. repetition: Prevents excessive pattern repetition
   - alpha/token based detection
   - max setting for allowed repetitions
   - Required fields: max

7. conditional: Pattern-based conditional checks
   - first/second patterns with logical operators
   - Ensures one pattern implies another
   - Required fields: first, second
   - Optional: exceptions array

8. metric: Checks readability metrics
   - formula options: Flesch-Kincaid, etc.
   - Required fields: formula

9. sequence: Validates pattern order
   - Ensures proper sequence of elements
   - Required fields: pattern

10. script: Run custom scripts for complex validation
    - Required fields: script (path to script file)

OUTPUT FORMAT:
You must provide exactly 3 rule alternatives in the following JSON format:

```json
{{
  "rules": [
    {{
      "rule": "YAML rule content here",
      "confidence": 85,
      "reasoning": "Brief explanation of approach and confidence level"
    }},
    {{
      "rule": "Alternative YAML rule content",
      "confidence": 70,
      "reasoning": "Brief explanation of this alternative approach"
    }},
    {{
      "rule": "Third alternative YAML rule content", 
      "confidence": 60,
      "reasoning": "Brief explanation of this third approach"
    }}
  ]
}}
```

CONFIDENCE SCORING GUIDELINES:
- 90-100: Perfect match with clear examples in context, standard Vale pattern
- 80-89: Good match with some examples, reliable approach
- 70-79: Reasonable approach but less certain about effectiveness
- 60-69: Experimental approach, may need refinement
- Below 60: Avoid using

CONTEXT:
{context}

REQUEST:
{request}

SEVERITY:
{severity}

JSON OUTPUT:
"""

rule_prompt = PromptTemplate(
    input_variables=["context", "request", "severity"],
    template=RULE_PROMPT_TEMPLATE
)

EXPLANATION_PROMPT_TEMPLATE = """
SYSTEM:
You are a helpful assistant. Your task is to explain how a Vale linter rule was generated.
Based on the original user request, the retrieved documentation and rule examples (from Vale core, Google, and Microsoft style guides), and the final YAML rule, explain in a few sentences:
1. Why you chose the specific `extends` value.
2. How the retrieved context (documentation and/or example rules) helped you construct the rule.
3. If you used any specific example rules as templates, mention which style guide they came from.

USER REQUEST:
{request}

RETRIEVED CONTEXT:
{context_str}

GENERATED RULE:
```yaml
{rule}
```

EXPLANATION:
"""

explanation_prompt = PromptTemplate(
    input_variables=["request", "context_str", "rule"],
    template=EXPLANATION_PROMPT_TEMPLATE
)

def smart_retriever_func(query):
    """
    Enhanced retriever that prioritizes rule examples and balances content types.
    """
    # Get initial results using current base retriever
    docs = current_base_retriever.invoke(query)
    
    # Separate rule examples from documentation
    rule_docs = [doc for doc in docs if doc.metadata.get('file_type') == 'yaml_rule']
    doc_docs = [doc for doc in docs if doc.metadata.get('file_type') != 'yaml_rule']
    
    # Prioritize rule examples (aim for 60% rule examples, 40% docs)
    final_docs = []
    
    # Add up to 6 rule examples
    final_docs.extend(rule_docs[:6])
    
    # Fill remaining slots with documentation
    remaining_slots = max(0, 8 - len(final_docs))
    final_docs.extend(doc_docs[:remaining_slots])
    
    return final_docs

# Create a LangChain-compatible retriever
smart_retriever = RunnableLambda(smart_retriever_func)

def format_docs(docs):
    formatted_parts = []
    
    for doc in docs:
        if doc.metadata.get('file_type') == 'yaml_rule':
            # Format rule examples differently with style guide info
            rule_name = doc.metadata.get('rule_name', 'Unknown')
            rule_type = doc.metadata.get('rule_type', 'unknown')
            style_guide = doc.metadata.get('style_guide', 'unknown')
            style_guide_display = style_guide.upper() if style_guide != 'vale-core' else 'VALE'
            formatted_parts.append(f"EXAMPLE RULE ({rule_type}) from {style_guide_display}: {rule_name}\n{doc.page_content}")
        else:
            # Regular documentation
            source = doc.metadata.get('source', 'Unknown')
            formatted_parts.append(f"DOCUMENTATION: {source}\n{doc.page_content}")
    
    return "\n\n" + "="*50 + "\n\n".join(formatted_parts)

def get_rule_gen_chain():
    """Get rule generation chain with current LLM."""
    return (
        RunnablePassthrough.assign(context_str=lambda x: format_docs(x["context"]))
        | rule_prompt
        | current_llm
        | StrOutputParser()
    )

def get_explanation_chain():
    """Get explanation chain with current LLM."""
    return explanation_prompt | current_llm | StrOutputParser()

def get_rag_chain():
    """Get RAG chain with current components."""
    return (
        RunnablePassthrough.assign(
            context=itemgetter("request") | smart_retriever
        ).assign(
            rule=get_rule_gen_chain()
        )
    )

# Initialize chains
rag_chain = get_rag_chain()


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get available AI providers based on configured API keys."""
    try:
        providers = get_available_providers()
        return jsonify({
            'providers': providers,
            'current': AI_PROVIDER,
            'count': len(providers)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current AI provider and model information."""
    try:
        chat_model = get_chat_model()
        embedding_model = get_embedding_model()
        
        if AI_PROVIDER == "gemini":
            model_name = chat_model.model
        elif AI_PROVIDER == "openai":
            model_name = chat_model.model_name
        elif AI_PROVIDER == "claude":
            model_name = chat_model.model
        else:
            model_name = "Unknown"
            
        return jsonify({
            'provider': AI_PROVIDER.upper(),
            'chat_model': model_name,
            'embedding_model': type(embedding_model).__name__,
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch-provider', methods=['POST'])
def switch_provider_endpoint():
    """Switch AI provider."""
    try:
        data = request.get_json()
        new_provider = data.get('provider')
        
        if not new_provider:
            return jsonify({'error': 'Provider is required'}), 400
        
        # Switch provider and reinitialize
        switch_provider(new_provider)
        
        # Reinitialize global chains
        global rag_chain
        rag_chain = get_rag_chain()
        
        return jsonify({
            'success': True,
            'provider': AI_PROVIDER.upper(),
            'message': f'Switched to {AI_PROVIDER.upper()} provider'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-rule', methods=['POST'])
def generate_rule():
    import json
    import re
    
    data = request.get_json()
    user_request = data.get('request')
    severity = data.get('severity', 'warning')

    if not user_request:
        return jsonify({'error': 'Missing required field: request'}), 400

    try:
        # First, invoke the main chain to get the rules and context
        rag_result = rag_chain.invoke({"request": user_request, "severity": severity})
        
        # Parse the JSON response containing multiple rules
        rules_response = rag_result.get('rule', '{}')
        
        # Extract JSON from potential code blocks or find raw JSON
        json_str = None
        
        # Try to extract from ```json code blocks first
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', rules_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks (look for the complete structure)
            json_match = re.search(r'(\{\s*"rules"\s*:\s*\[.*?\]\s*\})', rules_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'(\{.*?\})', rules_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
        
        if json_str:
            try:
                parsed_rules = json.loads(json_str)
                rules_list = parsed_rules.get('rules', [])
                
                # Validate that we have valid rules
                if not rules_list or not isinstance(rules_list, list):
                    raise ValueError("Invalid rules structure")
                    
                # Ensure each rule has required fields
                for rule in rules_list:
                    if not isinstance(rule, dict) or 'rule' not in rule:
                        raise ValueError("Invalid rule format")
                    
                    # Set default values for missing fields
                    rule['confidence'] = rule.get('confidence', 75)
                    rule['reasoning'] = rule.get('reasoning', 'No reasoning provided')
                        
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {rules_response[:500]}...")
                # Fallback if JSON parsing fails
                rules_list = [{
                    "rule": rules_response,
                    "confidence": 70,
                    "reasoning": "JSON parsing failed, using raw response"
                }]
        else:
            # No JSON found, treat as single rule
            rules_list = [{
                "rule": rules_response,
                "confidence": 75,
                "reasoning": "No JSON structure found, using raw response"
            }]

        # Generate explanation for the primary rule
        primary_rule = rules_list[0]['rule'] if rules_list else rules_response
        explanation_input = {
            "request": user_request,
            "context_str": format_docs(rag_result["context"]),
            "rule": primary_rule
        }
        explanation = get_explanation_chain().invoke(explanation_input)

        # Format the debug context for display
        debug_context = []
        for doc in rag_result.get("context", []):
            source = doc.metadata.get('source', 'Unknown')
            source = os.path.relpath(source, os.path.join(os.path.dirname(__file__), "vale_sh_repo"))
            debug_context.append(f"Source: {source}\n---\n{doc.page_content}")
        debug_context_str = "\n\n==================================================\n\n".join(debug_context)

        return jsonify({
            'rules': rules_list,
            'debug_context': debug_context_str,
            'explanation': explanation
        })

    except Exception as e:
        error_message = f"An error occurred during rule generation: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
