
import os
import sys
import git
import tempfile
import shutil
from tqdm import tqdm

import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Load environment variables
load_dotenv()

# --- Configuration ---
REPO_URL = "https://github.com/errata-ai/vale.sh.git"
REPO_PATH = os.path.join(os.path.dirname(__file__), "vale_sh_repo")
DOCS_PATH = os.path.join(REPO_PATH, "src/lib/content")
VALE_REPO_PATH = os.path.join(os.path.dirname(__file__), "vale_repo")
RULE_EXAMPLES_PATH = os.path.join(VALE_REPO_PATH, "testdata/styles")

# External style repositories
GOOGLE_REPO_URL = "https://github.com/errata-ai/Google.git"
MICROSOFT_REPO_URL = "https://github.com/errata-ai/Microsoft.git"

DB_BASE_PATH = os.path.join(os.path.dirname(__file__), "vectordb")

def get_db_path_for_provider(provider):
    """Get database path specific to the provider's embedding model."""
    if provider == "gemini":
        return os.path.join(DB_BASE_PATH, "gemini_768d")
    elif provider in ["openai", "claude"]:
        return os.path.join(DB_BASE_PATH, "openai_3072d")
    else:
        return DB_BASE_PATH
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# AI Provider configuration
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

def get_embedding_model_for_provider(provider):
    """Initialize embeddings model for a specific provider."""
    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    elif provider in ["openai", "claude"]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI/Claude provider") 
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def get_available_providers():
    """Get list of available providers based on configured API keys."""
    providers = []
    
    if GEMINI_API_KEY:
        providers.append('gemini')
    if OPENAI_API_KEY:
        providers.append('openai')
    if ANTHROPIC_API_KEY and OPENAI_API_KEY:
        providers.append('claude')
    
    return providers

class TqdmProgress(git.remote.RemoteProgress):
    """
    Helper class to show a progress bar during git clone.
    """
    def __init__(self):
        super().__init__()
        self.pbar = None

    def update(self, op_code, cur_count, max_count=None, message=''):
        if not self.pbar:
            self.pbar = tqdm(total=max_count, unit='B', unit_scale=True)
        
        self.pbar.set_description(message[:20].ljust(20))
        self.pbar.n = cur_count
        self.pbar.refresh()
        
        if cur_count == max_count:
            self.pbar.close()


def load_yaml_rules(rules_path):
    """
    Load Vale rule YAML files and convert them to Documents with rich metadata.
    """
    documents = []
    
    for root, dirs, files in os.walk(rules_path):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Try to fix common YAML issues before parsing
                    try:
                        rule_data = yaml.safe_load(content)
                    except yaml.YAMLError as yaml_error:
                        # Try to fix common escape sequence issues
                        if "unknown escape character" in str(yaml_error):
                            # Fix \' inside double quotes by replacing with just '
                            fixed_content = content.replace("\\'", "'")
                            try:
                                rule_data = yaml.safe_load(fixed_content)
                            except yaml.YAMLError:
                                # If still fails, try other common fixes
                                # Replace problematic escape sequences
                                fixed_content = content.replace("\\\\", "\\")
                                rule_data = yaml.safe_load(fixed_content)
                        else:
                            raise yaml_error
                    
                    if rule_data and isinstance(rule_data, dict):
                        # Extract rule type from 'extends' field
                        rule_type = rule_data.get('extends', 'unknown')
                        
                        # Create descriptive content that includes both the YAML and explanation
                        yaml_content = yaml.dump(rule_data, default_flow_style=False, sort_keys=False)
                        
                        # Create a rich description of the rule
                        description_parts = [
                            f"Rule Type: {rule_type}",
                            f"Rule Name: {os.path.splitext(file)[0]}",
                            f"Message: {rule_data.get('message', 'No message specified')}",
                            f"Level: {rule_data.get('level', 'warning')}"
                        ]
                        
                        # Add type-specific information
                        if rule_type == 'substitution' and 'swap' in rule_data:
                            description_parts.append(f"Substitutions: {len(rule_data['swap'])} replacements defined")
                        elif rule_type == 'existence' and 'tokens' in rule_data:
                            description_parts.append(f"Tokens: {len(rule_data['tokens'])} patterns to detect")
                        elif rule_type == 'occurrence' and 'max' in rule_data:
                            description_parts.append(f"Maximum occurrences: {rule_data['max']}")
                        elif rule_type == 'metric' and 'formula' in rule_data:
                            description_parts.append("Contains readability metric formula")
                        
                        description = "\n".join(description_parts)
                        
                        # Combine description with YAML for embedding
                        content = f"{description}\n\nYAML Rule Definition:\n{yaml_content}"
                        
                        # Get relative path for source
                        rel_path = os.path.relpath(file_path, rules_path)
                        
                        # Determine rule source/style guide
                        style_guide = 'vale-core'
                        if 'google' in rules_path.lower():
                            style_guide = 'google'
                        elif 'microsoft' in rules_path.lower():
                            style_guide = 'microsoft'
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'rule_name': os.path.splitext(file)[0],
                                'rule_type': rule_type,
                                'level': rule_data.get('level', 'warning'),
                                'file_type': 'yaml_rule',
                                'category': os.path.dirname(rel_path),
                                'message': rule_data.get('message', ''),
                                'style_guide': style_guide
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
                    continue
    
    return documents


def clone_external_repos_temp():
    """
    Clone external style repositories to temporary directories.
    Returns paths to the cloned repositories or None if failed.
    """
    temp_dir = tempfile.mkdtemp(prefix="valegen_styles_")
    
    repos_to_clone = [
        (GOOGLE_REPO_URL, "google", "Google style guide"),
        (MICROSOFT_REPO_URL, "microsoft", "Microsoft style guide")
    ]
    
    cloned_paths = {}
    
    for repo_url, repo_name, description in repos_to_clone:
        repo_path = os.path.join(temp_dir, repo_name)
        print(f"\nCloning {description} to temporary directory...")
        try:
            git.Repo.clone_from(repo_url, repo_path, progress=TqdmProgress())
            cloned_paths[repo_name] = repo_path
            print(f"✓ {description} cloned successfully to temp directory.")
        except Exception as e:
            print(f"Warning: Failed to clone {description}: {e}")
            print(f"Continuing without {description}...")
    
    return temp_dir, cloned_paths


def main():
    """
    Main function to run the setup process.
    """
    print("--- Starting Vale Rule Generator Setup ---")

    # 1. Check for available providers and API keys
    available_providers = get_available_providers()
    
    if not available_providers:
        print("\nERROR: No API keys found. Please set at least one of:")
        print("  - GEMINI_API_KEY for Google Gemini")
        print("  - OPENAI_API_KEY for OpenAI GPT-4o")
        print("  - ANTHROPIC_API_KEY + OPENAI_API_KEY for Claude")
        sys.exit(1)
    
    print(f"Found API keys for providers: {', '.join(available_providers)}")
    
    # Show details about detected API keys
    print("\nDetected API keys:")
    if GEMINI_API_KEY:
        print(f"  ✓ GEMINI_API_KEY: {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 12 else '***'}")
    else:
        print("  ✗ GEMINI_API_KEY: Not set")
    
    if OPENAI_API_KEY:
        print(f"  ✓ OPENAI_API_KEY: {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 12 else '***'}")
    else:
        print("  ✗ OPENAI_API_KEY: Not set")
    
    if ANTHROPIC_API_KEY:
        print(f"  ✓ ANTHROPIC_API_KEY: {ANTHROPIC_API_KEY[:8]}...{ANTHROPIC_API_KEY[-4:] if len(ANTHROPIC_API_KEY) > 12 else '***'}")
    else:
        print("  ✗ ANTHROPIC_API_KEY: Not set")
    
    # Validate each provider's embedding model
    for provider in available_providers:
        try:
            get_embedding_model_for_provider(provider)
            print(f"✓ {provider.upper()} embedding model validated.")
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    # 2. Clone Vale repository if it doesn't exist
    if not os.path.exists(REPO_PATH):
        print(f"\nCloning Vale repository from {REPO_URL}...")
        try:
            git.Repo.clone_from(REPO_URL, REPO_PATH, progress=TqdmProgress())
            print("✓ Repository cloned successfully.")
        except Exception as e:
            print(f"\nERROR: Failed to clone repository: {e}")
            sys.exit(1)
    else:
        print(f"\nVale repository already exists at {REPO_PATH}. Skipping clone.")

    # 3. Clone external style repositories to temporary directories
    temp_dir, external_repos = clone_external_repos_temp()

    # 4. Load documentation
    print("\nLoading documentation from a local directory...")
    try:
        loader = DirectoryLoader(
            DOCS_PATH, 
            glob="**/*.md", 
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} documentation files.")
    except Exception as e:
        print(f"\nERROR: Failed to load documents: {e}")
        sys.exit(1)

    # 5. Load Vale rule examples
    print("\nLoading Vale rule examples...")
    try:
        if os.path.exists(RULE_EXAMPLES_PATH):
            rule_documents = load_yaml_rules(RULE_EXAMPLES_PATH)
            documents.extend(rule_documents)
            print(f"✓ Loaded {len(rule_documents)} core rule examples.")
        else:
            print(f"Warning: Rule examples path not found at {RULE_EXAMPLES_PATH}")
            print("Core rule examples will not be included in the embeddings.")
    except Exception as e:
        print(f"Warning: Failed to load core rule examples: {e}")
        print("Continuing without core rule examples...")

    # 6. Load external style rules from temporary directories
    print("\nLoading external style guide rules...")
    external_rule_count = 0
    
    try:
        # Load Google style rules
        if 'google' in external_repos:
            google_rules_path = os.path.join(external_repos['google'], "Google")
            if os.path.exists(google_rules_path):
                try:
                    google_rules = load_yaml_rules(google_rules_path)
                    documents.extend(google_rules)
                    external_rule_count += len(google_rules)
                    print(f"✓ Loaded {len(google_rules)} Google style rules.")
                except Exception as e:
                    print(f"Warning: Failed to load Google style rules: {e}")
            else:
                print(f"Warning: Google style rules not found at {google_rules_path}")
        
        # Load Microsoft style rules
        if 'microsoft' in external_repos:
            microsoft_rules_path = os.path.join(external_repos['microsoft'], "Microsoft")
            if os.path.exists(microsoft_rules_path):
                try:
                    microsoft_rules = load_yaml_rules(microsoft_rules_path)
                    documents.extend(microsoft_rules)
                    external_rule_count += len(microsoft_rules)
                    print(f"✓ Loaded {len(microsoft_rules)} Microsoft style rules.")
                except Exception as e:
                    print(f"Warning: Failed to load Microsoft style rules: {e}")
            else:
                print(f"Warning: Microsoft style rules not found at {microsoft_rules_path}")
        
        if external_rule_count > 0:
            print(f"✓ Total external style rules loaded: {external_rule_count}")
        else:
            print("Warning: No external style rules were loaded.")
            
    except Exception as e:
        print(f"Warning: Error processing external style rules: {e}")

    # 7. Split documents into chunks
    print(f"\nSplitting {len(documents)} total documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"✓ Documents split into {len(docs)} chunks.")

    # 8. Create embeddings and store in ChromaDB for each provider
    print(f"\nCreating embeddings for all available providers...")
    print("This may take a few minutes depending on the number of documents...")
    
    # Group providers by embedding type to avoid duplicate work
    embedding_groups = {}
    for provider in available_providers:
        if provider == "gemini":
            embedding_groups["gemini"] = ["gemini"]
        elif provider in ["openai", "claude"]:
            if "openai" not in embedding_groups:
                embedding_groups["openai"] = []
            embedding_groups["openai"].append(provider)
    
    try:
        for embedding_type, providers in embedding_groups.items():
            print(f"\nProcessing {embedding_type.upper()} embeddings for: {', '.join(providers)}")
            
            # Get the embedding model for this type
            embedding_provider = providers[0]  # Use first provider as representative
            embeddings = get_embedding_model_for_provider(embedding_provider)
            db_path = get_db_path_for_provider(embedding_provider)
            
            # Create the vector store
            vectordb = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            # Add documents with progress bar
            for i in tqdm(range(0, len(docs), 32), desc=f"Embedding for {embedding_type}"):
                batch = docs[i:i+32]
                vectordb.add_documents(documents=batch)
            
            print(f"✓ Vector database created at {db_path}")
            print(f"  Available for providers: {', '.join(providers)}")

    except Exception as e:
        print(f"\nERROR: Failed to create vector database: {e}")
        # Clean up all directories even on error
        try:
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory")
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up temp directory: {cleanup_error}")
        
        try:
            if os.path.exists(REPO_PATH):
                shutil.rmtree(REPO_PATH)
                print(f"✓ Cleaned up Vale repository")
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up Vale repository: {cleanup_error}")
        
        try:
            if os.path.exists(VALE_REPO_PATH):
                shutil.rmtree(VALE_REPO_PATH)
                print(f"✓ Cleaned up Vale core repository")
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up Vale core repository: {cleanup_error}")
        
        sys.exit(1)

    # 9. Clean up temporary directories and repositories
    print(f"\nCleaning up temporary directories and repositories...")
    
    # Clean up external style repositories
    try:
        shutil.rmtree(temp_dir)
        print(f"✓ Temporary style repositories removed from {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory: {e}")
    
    # Clean up Vale repository
    try:
        if os.path.exists(REPO_PATH):
            shutil.rmtree(REPO_PATH)
            print(f"✓ Vale repository removed from {REPO_PATH}")
    except Exception as e:
        print(f"Warning: Failed to clean up Vale repository: {e}")
    
    # Clean up Vale core repository if it exists
    try:
        if os.path.exists(VALE_REPO_PATH):
            shutil.rmtree(VALE_REPO_PATH)
            print(f"✓ Vale core repository removed from {VALE_REPO_PATH}")
    except Exception as e:
        print(f"Warning: Failed to clean up Vale core repository: {e}")

    print("\n--- Setup Complete! ---")
    print("You can now run the main application.")

if __name__ == "__main__":
    main()
