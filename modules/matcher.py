"""
Deterministic, explainable CV ↔ JD skill matching.

A JD requirement is satisfied when the CV shows:
  * the skill itself, under any alias/abbreviation      → "direct"
    ("NLP" ⇔ "Natural Language Processing" are one canonical skill)
  * a more specific technology that implies it           → "inferred"
    ("MongoDB" ⇒ "NoSQL", "CNN" ⇒ "Neural Networks", "BERT" ⇒ "NLP")
  * all components of a composite stack                  → "inferred"
    (MongoDB + Express + React + Node ⇒ "MERN")

This gives semantic matching without depending on the exact JD wording,
while every match stays reproducible and comes with a CV quote. The
optional LLM pass (llm_engine.py) is only consulted for requirements
still unmatched here.
"""
import re
from dataclasses import dataclass

from config import INFERENCE_CREDIT, MATCH_CREDIT
from modules.skill_extractor import (
    _NOT_REST_LOOKBEHIND, RAG_LLM_SKILLS, RAG_RETRIEVAL_SKILLS, canonicalize_skill, skill_mentions,
)

ISA, STRONG, MODERATE, WEAK = "isa", "strong", "moderate", "weak"


def _e(kind: str, *sources) -> list:
    return [(src, kind) for src in sources]


_CNN_ARCHITECTURES = (r"re:\b(?:res[\s-]?net|vgg[\s-]?\d*|u[\s-]?net|efficient[\s-]?net|mobile[\s-]?net|"
                      r"inception[\s-]?v\d|alex[\s-]?net|dense[\s-]?net|(?:mask|faster)[\s-]?r[\s-]?cnn|"
                      r"convolutional)\b")
_VECTOR_STORES = ("pinecone", "chromadb", "faiss", "qdrant", "weaviate", "milvus", "pgvector", "lancedb")
_UNIT_TEST_TOOLS = ("pytest", "jest", "junit", "unittest", "mocha", "vitest", "testing library")
_E2E_TEST_TOOLS = ("selenium", "cypress", "playwright", "appium")
_ORMS = ("hibernate", "jpa", "sqlalchemy", "prisma", "sequelize", "typeorm", "mongoose", "drizzle")
_BI_TOOLS = ("power bi", "tableau", "looker", "qlik")
_LLM_AGENT_FRAMEWORKS = ("crewai", "autogen")
_AWS_SERVICES = (r"re:(?-i:\b(?:EC2|S3|EKS|ECS|RDS|CloudWatch|CloudFront|SQS|SNS|Fargate|Bedrock|"
                 r"Elastic\s+Beanstalk)\b)|\bAWS\s+Lambda\b")
_MANAGED_K8S = r"re:(?-i:\b(?:EKS|AKS|GKE)\b)"

IMPLIED_BY = {
    "machine learning": _e(STRONG, "deep learning", "scikit-learn", "xgboost", "lightgbm", "catboost",
                           "supervised learning", "unsupervised learning", "neural networks", "tensorflow",
                           "pytorch", "keras", "predictive modeling", "feature engineering", "mlops",
                           "reinforcement learning")
                        + _e(MODERATE, "classification", "regression", "computer vision",
                             "natural language processing"),
    "deep learning": _e(STRONG, "neural networks", "cnn", "rnn", "lstm", "gan", "transformers", "tensorflow",
                        "pytorch", "keras", "yolo", "bert", "autoencoders", "lora", "stable diffusion", "diffusers",
                        _CNN_ARCHITECTURES)
                     + _e(MODERATE, "fine-tuning", "huggingface"),
    "neural networks": _e(ISA, "cnn", "rnn", "lstm", "gan", "autoencoders", r"re:\bneural\s+net",
                          r"re:\bmulti[\s-]?layer\s+perceptron")
                       + _e(STRONG, "deep learning", "transformers", "bert", "yolo", "tensorflow", "pytorch",
                            "keras", _CNN_ARCHITECTURES),
    "cnn": _e(ISA, _CNN_ARCHITECTURES)
           + _e(STRONG, "yolo")
           + _e(MODERATE, "image classification", "object detection", "image segmentation"),
    "supervised learning": _e(STRONG, "classification", "regression", "text classification",
                              "image classification", "object detection", "sentiment analysis",
                              "xgboost", "lightgbm", "catboost",
                              r"re:\b(?:classifier|classification|regression)\s+models?\b",
                              r"re:\btrain(?:ed|ing)\b.{0,60}\b(?:classifier|labell?ed|regression)",
                              r"re:\b(?:random\s+forests?|decision\s+trees?|logistic\s+regression|"
                              r"support\s+vector\s+machines?|svms?)\b")
                           + _e(MODERATE, "scikit-learn"),
    "unsupervised learning": _e(STRONG, "clustering", r"re:\b(?:k-?means|dbscan|pca|dimensionality\s+reduction)\b"),
    "natural language processing": _e(STRONG, "bert", "spacy", "nltk", "sentiment analysis",
                                      "named entity recognition", "text classification",
                                      r"re:\b(?:text|language)\s+(?:mining|processing|models?|analytics)\b")
                                   + _e(MODERATE, "llm", "gpt", "transformers", "rag", "chatbot", "chatbots",
                                        "langchain", "huggingface", "speech recognition"),
    "computer vision": _e(STRONG, "yolo", "object detection", "image segmentation", "image classification",
                          "face recognition", _CNN_ARCHITECTURES)
                       + _e(MODERATE, "opencv", "image processing", "ocr", "cnn"),
    "artificial intelligence": _e(STRONG, "machine learning", "deep learning", "natural language processing",
                                  "computer vision", "llm", "generative ai", "neural networks", "ai agents"),
    "predictive modeling": _e(ISA, r"re:\bpredictive\s+(?:model|analy)")
                           + _e(MODERATE, "xgboost", "lightgbm", "catboost", "time series",
                                r"re:\bforecast(?:ing|ed)?\s+(?:models?|demand|sales|revenue|churn)"),
    "transformers": _e(STRONG, "bert", "gpt", "llama", "mistral") + _e(MODERATE, "huggingface"),
    "huggingface": _e(ISA, "transformers"),
    "object detection": _e(ISA, "yolo", r"re:\b(?:faster[\s-]?r[\s-]?cnn|detectron2?|ssd\s+detector)\b"),
    "fine-tuning": _e(ISA, "lora", "peft", r"re:\bfine[\s-]?tun")
                   + _e(MODERATE, r"re:\btransfer\s+learning\b"),
    "mlops": _e(STRONG, "mlflow", "kubeflow") + _e(MODERATE, "dvc", "wandb", "sagemaker"),

    "generative ai": _e(STRONG, "llm", "gpt", "rag", "stable diffusion", "diffusers", "gan", "ai agents",
                        "langgraph", "ollama", "vllm", "prompt engineering", *_LLM_AGENT_FRAMEWORKS)
                     + _e(MODERATE, "openai", "gemini", "claude", "langchain", "llamaindex")
                     + _e(WEAK, "huggingface"),
    "llm": _e(ISA, "gpt", "llama", "mistral")
           + _e(STRONG, "rag", "langchain", "llamaindex", "langgraph", "ollama", "vllm", "prompt engineering",
                "ai agents", "function calling", *_LLM_AGENT_FRAMEWORKS, r"re:\blarge\s+language\s+models?\b")
           + _e(MODERATE, "openai", "gemini", "claude", "mcp")
           + _e(WEAK, "huggingface"),
    "prompt engineering": _e(ISA, r"re:\bprompt(?:s|ing)?\s+(?:design|engineer\w*|templat\w*|optimi[sz]\w*|tuning)",
                             r"re:\b(?:system|few[\s-]shot|zero[\s-]shot|chain[\s-]of[\s-]thought)\s+prompt")
                          + _e(MODERATE, "rag", "langchain", "llamaindex", "langgraph", "ai agents",
                               *_LLM_AGENT_FRAMEWORKS)
                          + _e(WEAK, "llm", "openai", "gpt", "gemini", "claude", "ollama"),
    "ai agents": _e(ISA, *_LLM_AGENT_FRAMEWORKS,
                    r"re:\b(?:ai|llm|autonomous|multi[\s-]?agent|langchain|langgraph|conversational|voice)"
                    r"\s+agents?\b|\bagentic\b")
                 + _e(STRONG, "langgraph")
                 + _e(MODERATE, "function calling", "mcp"),
    "function calling": _e(ISA, r"re:\b(?:tool|function)[\s-]?(?:calling|calls|use)\b")
                        + _e(MODERATE, "mcp", "ai agents", *_LLM_AGENT_FRAMEWORKS),
    "rag": _e(ISA, r"re:\bretrieval[\s-]+augmented\s+generat", r"re:\bretrieval\b.{0,40}\bgenerat")
           + _e(STRONG, r"re:\bsemantic\s+search\b.{0,80}\b(?:llm|gpt|gemini|claude|llama)"),
    "vector database": _e(ISA, *_VECTOR_STORES) + _e(MODERATE, "rag"),
    "vector search": _e(ISA, "vector database", *_VECTOR_STORES, "semantic search")
                     + _e(STRONG, "rag")
                     + _e(WEAK, "embeddings"),
    "semantic search": _e(STRONG, "vector database", "vector search", *_VECTOR_STORES) + _e(MODERATE, "rag"),
    "embeddings": _e(STRONG, "vector database", "vector search", *_VECTOR_STORES, r"re:\bembedding"),
    "openai": _e(STRONG, "gpt"),

    "python": _e(STRONG, "django", "flask", "fastapi", "pandas", "numpy", "pytorch", "scikit-learn", "keras",
                 "tensorflow", "scipy", "matplotlib", "seaborn", "streamlit", "pyspark", "jupyter", "sqlalchemy",
                 "pytest", "celery")
              + _e(MODERATE, "langchain", "opencv", "yolo", "airflow"),
    "java": _e(STRONG, "spring boot", "spring", "hibernate", "j2ee", "maven", "junit"),
    "javascript": _e(STRONG, "typescript", "react", "angular", "vue", "nodejs", "jquery", "next.js", "express",
                     "react native", "nuxt.js", "svelte", "nestjs"),
    "typescript": _e(MODERATE, "angular", "nestjs"),
    "dart": _e(STRONG, "flutter"),
    "swift": _e(STRONG, "swiftui"),
    "kotlin": _e(STRONG, "jetpack compose"),
    "php": _e(STRONG, "laravel", "codeigniter"),
    "c#": _e(STRONG, "asp.net"),

    "rest api": _e(STRONG, "fastapi", "django rest framework")
                + _e(MODERATE, "express", "flask", "spring boot", "postman", "swagger", "nestjs", "laravel"),
    "api development": _e(ISA, "rest api", "graphql")
                       + _e(STRONG, "fastapi", "django rest framework")
                       + _e(MODERATE, "express", "flask", "spring boot", "nestjs"),
    "html": _e(STRONG, "react", "angular", "vue", "bootstrap", "tailwind"),
    "css": _e(STRONG, "bootstrap", "tailwind", "sass", "less", "material ui"),
    "nodejs": _e(STRONG, "express", "nestjs"),
    "react": _e(STRONG, "next.js", "react native"),
    "mvc": _e(ISA, "spring mvc")
           + _e(MODERATE, "laravel", "codeigniter", "asp.net", "django"),
    "mvvm": _e(ISA, r"re:\bmvvm\b"),
    "orm": _e(ISA, *_ORMS) + _e(MODERATE, "django", "laravel", "spring boot"),
    "json": _e(ISA, r"re:\bjson\b") + _e(MODERATE, "rest api"),
    "microservices": _e(ISA, r"re:\bmicro[\s-]?services?\b") + _e(MODERATE, "grpc"),
    "message queues": _e(ISA, "kafka", "rabbitmq", r"re:(?-i:\b(?:SQS|ActiveMQ|ZeroMQ|NATS)\b)|\bpub[\s/-]?sub\b")
                      + _e(MODERATE, "celery"),
    "event-driven architecture": _e(STRONG, "kafka") + _e(MODERATE, "rabbitmq", "message queues", "webhooks"),
    "full stack": _e(ISA, "mern", "mean", "lamp"),
    "apache": _e(ISA, "lamp") + _e(MODERATE, r"re:\bxa+m+p+\b|\bwamp(?:server)?\b"),
    "spring": _e(ISA, "spring boot", "spring mvc", "spring security"),
    "j2ee": _e(MODERATE, "spring", "spring boot", "hibernate", "jpa", "tomcat"),
    "backend development": _e(ISA, "django", "flask", "fastapi", "nodejs", "express", "laravel", "spring boot",
                              "nestjs", "spring", "j2ee", "codeigniter", "asp.net",
                              r"re:\bback[\s-]?end\b", r"re:\bserver[\s-]side\b")
                           + _e(STRONG, "rest api", "api development", "microservices", "graphql")
                           + _e(MODERATE, "firebase", "supabase"),
    "frontend development": _e(ISA, "react", "angular", "vue", "next.js", "svelte", r"re:\bfront[\s-]?end\b")
                            + _e(STRONG, "html", "css", "javascript", "typescript", "tailwind", "bootstrap"),
    "ui/ux": _e(STRONG, "figma", "adobe xd")
             + _e(MODERATE, r"re:\bwire[\s-]?fram", r"re:\bprototyp",
                  "frontend development", "tailwind", "bootstrap", "material ui")
             + _e(WEAK, r"re:\bresponsive\s+(?:design|layouts?|ui|web)"),

    "mobile development": _e(ISA, "flutter", "react native", "swiftui", "jetpack compose", "android", "ios",
                             "xamarin", "ionic", "android sdk", "cross-platform development",
                             r"re:\bmobile\s+app(?:lication)?s?\b")
                          + _e(MODERATE, "kotlin", "swift"),
    "cross-platform development": _e(ISA, "flutter", "react native", "xamarin", "ionic"),
    "android": _e(STRONG, "kotlin", "android studio", "android sdk", "jetpack compose")
               + _e(MODERATE, "flutter", "react native"),
    "ios": _e(STRONG, "swift", "swiftui", "xcode", "objective-c") + _e(MODERATE, "flutter", "react native"),
    "state management": _e(ISA, "bloc", "getx", "riverpod", "provider", "redux", "zustand"),

    "nosql": _e(ISA, "mongodb", "redis", "cassandra", "dynamodb", "firebase", "couchbase", "neo4j",
                "elasticsearch"),
    "sql": _e(ISA, "mysql", "postgresql", "sql server", "sqlite", "oracle", "mariadb", "pl/sql", "bigquery",
              "snowflake", "redshift", "supabase", "stored procedures")
           + _e(STRONG, "query optimization"),
    "database": _e(ISA, "sql", "nosql", "mysql", "postgresql", "sql server", "sqlite", "oracle", "mariadb",
                   "mongodb", "redis", "cassandra", "dynamodb", "firebase", "supabase", "pl/sql", "neo4j",
                   "elasticsearch", "database design", "database administration")
                + _e(MODERATE, r"re:\bdatabases?\b"),

    "data engineering": _e(STRONG, "etl", "airflow", "dbt", "spark", "pyspark", "databricks", "data pipelines",
                           "data warehousing")
                        + _e(MODERATE, "kafka", "snowflake", "bigquery", "redshift", "hadoop", "flink"),
    "data pipelines": _e(ISA, "etl", r"re:\b(?:data|etl|elt|ingestion|streaming)\s+pipelines?\b")
                      + _e(STRONG, "airflow", "dbt", "databricks")
                      + _e(MODERATE, "spark", "pyspark", "kafka", "flink"),
    "etl": _e(STRONG, "dbt", "data pipelines") + _e(MODERATE, "airflow"),
    "data warehousing": _e(ISA, "snowflake", "redshift", "bigquery") + _e(MODERATE, "dbt"),
    "big data": _e(STRONG, "spark", "pyspark", "hadoop", "hive", "databricks", "flink")
                + _e(MODERATE, "kafka", "bigquery", "snowflake"),
    "spark": _e(ISA, "pyspark") + _e(STRONG, "databricks"),
    "business intelligence": _e(ISA, *_BI_TOOLS) + _e(MODERATE, "data warehousing"),
    "data visualization": _e(STRONG, *_BI_TOOLS, "matplotlib", "seaborn", "plotly", r"re:\bdashboards?\b"),
    "data science": _e(STRONG, "machine learning") + _e(MODERATE, "data analysis", "pandas", "statistics"),
    "data analysis": _e(STRONG, "pandas", "data analytics", "power bi", "tableau", r"re:\bdata\s+analy")
                     + _e(MODERATE, "numpy"),
    "statistics": _e(ISA, "statistical modeling", r"re:\bstatistic(?:s|al)\b"),

    "git": _e(ISA, "github", "gitlab", "bitbucket", "github actions"),
    "version control": _e(ISA, "git", "github", "gitlab", "svn", "bitbucket", "mercurial"),
    "ci/cd": _e(ISA, "github actions", "gitlab ci", "jenkins", "circleci", "travis ci", "argocd")
             + _e(STRONG, r"re:\b(?:build|deployment|release)\s+pipelines?\b"),
    "docker": _e(ISA, r"re:\bdocker(?:files?|[\s-]?compose|ized|hub)\b") + _e(MODERATE, "kubernetes"),
    "kubernetes": _e(ISA, "openshift", _MANAGED_K8S) + _e(STRONG, "helm", "argocd") + _e(MODERATE, "kubeflow"),
    "containerization": _e(ISA, "docker", "openshift", r"re:\bcontaineri[sz]") + _e(STRONG, "kubernetes"),
    "infrastructure as code": _e(ISA, "terraform", "cloudformation", "pulumi") + _e(STRONG, "ansible"),
    "devops": _e(MODERATE, "ci/cd", "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions",
                 "gitlab ci", "helm", "infrastructure as code"),
    "linux": _e(ISA, r"re:\b(?:ubuntu|centos|debian|red\s?hat|rhel|fedora|kali\s+linux)\b")
             + _e(MODERATE, "bash", "shell"),
    "aws": _e(ISA, "sagemaker", "dynamodb", "redshift", "cloudformation", _AWS_SERVICES),
    "azure": _e(ISA, "azure ml", r"re:(?-i:\bAKS\b)"),
    "google cloud": _e(ISA, "vertex ai", "bigquery", r"re:(?-i:\bGKE\b)|\bcloud\s+run\b"),
    "cloud": _e(ISA, "aws", "azure", "google cloud", "sagemaker", "vertex ai")
             + _e(MODERATE, "firebase", "serverless", r"re:\b(?:heroku|vercel|netlify|digitalocean)\b"),

    "unit testing": _e(ISA, *_UNIT_TEST_TOOLS, r"re:\bunit[\s-]?tests?\b") + _e(STRONG, "tdd"),
    "test automation": _e(ISA, *_E2E_TEST_TOOLS,
                          r"re:\bautomat\w*\s+(?:\w+\s+)?tests?\b|\btest(?:ing)?\s+automation\b")
                       + _e(MODERATE, *_UNIT_TEST_TOOLS),
    "software testing": _e(ISA, "unit testing", "test automation", "tdd", "bdd", *_UNIT_TEST_TOOLS,
                           *_E2E_TEST_TOOLS,
                           r"re:\b(?:manual|regression|integration|functional|qa|end[\s-]to[\s-]end)\s+test")
                        + _e(MODERATE, "postman"),

    "distributed systems": _e(STRONG, "microservices", r"re:\bdistributed\b")
                           + _e(MODERATE, "kafka", "kubernetes", "spark", "hadoop", "rabbitmq", "grpc"),
    "system design": _e(STRONG, "software architecture", "microservices",
                        r"re:\b(?:designed|architected)\b.{0,40}\b(?:system|architecture|platform|backend)"),
    "software architecture": _e(STRONG, "system design", "microservices", "clean architecture",
                                r"re:\barchitect(?:ed|ure)\b"),
    "performance optimization": _e(STRONG, "query optimization",
                                   r"re:\b(?:optimi[sz]ed|improv\w+)\b.{0,40}\b(?:performance|latency|speed|load\s+time|response\s+time)",
                                   r"re:\b(?:performance|latency|response\s+time)\b.{0,30}\b(?:optimi[sz]\w+|improve\w*|reduc\w+)"),
    "open source": _e(ISA, r"re:\bopen[\s-]source\b") + _e(MODERATE, r"re:\bcontribut\w+\b.{0,40}\b(?:github|repositor|librar)"),
    "agile": _e(ISA, "scrum", "kanban"),

    "crm": _e(ISA, "salesforce", "hubspot"),
    "sales": _e(STRONG, r"re:\b(?:closed|closing)\s+(?:\w+\s+){0,2}deals?\b|\bsales\s+(?:targets?|quotas?|pipelines?)\b"
                        r"|\b(?:met|exceeded|achieved)\s+(?:\w+\s+){0,2}(?:quotas?|targets?)\b")
             + _e(MODERATE, "account management", "business development", "lead generation"),
    "accounting": _e(STRONG, "bookkeeping", "quickbooks", "xero", "financial reporting")
                  + _e(MODERATE, "auditing", "taxation"),
    "bookkeeping": _e(STRONG, "quickbooks", "xero"),
    "financial reporting": _e(ISA, r"re:\bfinancial\s+(?:statements?|reports?)\b|\bbalance\s+sheets?\b"
                                   r"|\bprofit\s+(?:and|&)\s+loss\b|\bp\s?&\s?l\b")
                           + _e(MODERATE, "accounting"),
    "hris": _e(ISA, "bamboohr", "workday", "successfactors"),
    "hr operations": _e(STRONG, "hris") + _e(MODERATE, "payroll", "onboarding", "employee relations"),
    "compensation and benefits": _e(MODERATE, "payroll"),
    "recruitment": _e(STRONG, "linkedin recruiter",
                      r"re:\b(?:screened|interviewed|sourced|shortlisted|hired|recruited)\b(?:\s+\S+){0,5}?\s+"
                      r"(?:candidates?|applicants?|talent|staff|employees)\b",
                      r"re:\b(?:conducted|scheduled|coordinated|arranged)\s+(?:\w+\s+){0,2}interviews\b")
                   + _e(MODERATE, "ats", "boolean search", "employer branding"),
    "ms office": _e(ISA, "excel", "ms word", "powerpoint"),
    "upselling": _e(ISA, r"re:\bup[\s-]?sell", r"re:\bcross[\s-]?sell"),
    "customer success": _e(STRONG, "account management") + _e(MODERATE, "customer service"),
}


def implying_sources(skill: str) -> list:
    """Canonical skills (not regexes) that count as evidence for `skill`."""
    return [src for src, _ in IMPLIED_BY.get(skill, []) if not src.startswith("re:")]


def isa_instances(skill: str) -> set:
    """Canonical skills that ARE instances of `skill` (Excel is part of MS Office, BambooHR is an HRIS)."""
    return {src for src, kind in IMPLIED_BY.get(skill, []) if kind == ISA and not src.startswith("re:")}


CONCEPT_SKILLS = {
    "artificial intelligence", "machine learning", "deep learning", "neural networks", "supervised learning",
    "unsupervised learning", "reinforcement learning", "natural language processing", "computer vision",
    "generative ai", "llm", "rag", "vector search", "fine-tuning", "transfer learning", "prompt engineering",
    "feature engineering", "predictive modeling", "statistical modeling", "statistics", "pattern recognition",
    "data science", "data analysis", "data analytics", "data engineering", "data mining", "data visualization",
    "big data", "etl", "data warehousing", "database design", "query optimization", "time series",
    "object detection", "image processing", "image segmentation", "image classification", "face recognition",
    "ocr", "speech recognition", "recommendation systems", "sentiment analysis", "text classification",
    "rest api", "api development", "json", "microservices", "soa", "state management", "mvc", "mvvm", "mvp",
    "clean architecture", "oop", "tdd", "bdd", "agile", "scrum", "ci/cd", "devops", "mlops", "version control",
    "full stack", "cloud", "serverless", "embedded systems", "iot", "robotics", "sdk", "ai agents",
    "function calling", "semantic search", "quantization",
    "account management", "key account management", "sales", "business development", "lead generation",
    "negotiation", "customer service", "marketing", "digital marketing", "market research", "recruitment",
    "onboarding", "employee relations", "performance management", "budgeting", "forecasting",
    "financial analysis", "supply chain management", "procurement", "inventory management", "logistics",
    "project management", "event planning", "public relations", "content writing", "copywriting",
    "compensation and benefits", "labour law", "employer branding", "employee engagement",
    "training and development", "succession planning", "talent management", "hr operations", "hr policies",
    "upselling", "customer success", "backend development", "frontend development", "database",
    "distributed systems", "system design", "software architecture", "performance optimization", "open source",
    "mobile development", "cross-platform development", "unit testing", "test automation", "software testing",
    "infrastructure as code", "containerization", "orm", "data pipelines", "business intelligence",
    "message queues", "event-driven architecture", "ui/ux", "financial reporting", "bookkeeping",
}


def is_concept_requirement(req) -> bool:
    return any(s in CONCEPT_SKILLS for s in req.skills)


CONCEPT_HINTS = {
    "artificial intelligence": ["ai", "artificial", "intelligen", "machine learn", "ml", "deep learn", "neural",
                                "llm", "model"],
    "machine learning": ["machine learn", "ml", "model", "predict", "classif", "regress", "cluster", "train",
                         "scikit", "xgboost", "neural"],
    "deep learning": ["deep", "neural", "cnn", "rnn", "lstm", "transformer", "tensorflow", "pytorch", "keras"],
    "neural networks": ["neural", "cnn", "rnn", "lstm", "network", "deep", "transformer"],
    "supervised learning": ["supervis", "classif", "regress", "label", "predict", "train"],
    "unsupervised learning": ["unsupervis", "cluster", "kmeans", "k-means", "pca", "anomal"],
    "reinforcement learning": ["reinforc", "reward", "agent", "policy"],
    "natural language processing": ["nlp", "natural language", "text", "language model", "llm", "bert", "gpt",
                                    "sentiment", "chatbot", "token", "transformer", "spacy", "nltk"],
    "computer vision": ["vision", "image", "video", "opencv", "yolo", "detect", "segment", "face", "ocr", "cnn"],
    "generative ai": ["generat", "llm", "gpt", "diffusion", "gan", "openai", "gemini", "langchain"],
    "llm": ["llm", "gpt", "llama", "gemini", "claude", "openai", "language model", "langchain", "prompt"],
    "rag": ["rag", "retriev", "vector", "embedding", "langchain", "llamaindex", "knowledge base"],
    "vector search": ["vector", "embedding", "faiss", "pinecone", "chroma", "semantic search", "similarity"],
    "fine-tuning": ["fine", "tun", "lora", "peft", "train", "transfer"],
    "transfer learning": ["transfer", "pretrain", "fine"],
    "prompt engineering": ["prompt"],
    "feature engineering": ["feature"],
    "predictive modeling": ["predict", "forecast", "model"],
    "statistical modeling": ["statist", "regress", "hypothes", "probabil", "model"],
    "statistics": ["statist", "regress", "hypothes", "probabil"],
    "pattern recognition": ["pattern", "recogni", "classif"],
    "data science": ["data", "analy", "model", "statist"],
    "data analysis": ["data", "analy", "dashboard", "report", "insight", "excel", "power bi", "tableau", "pandas"],
    "data analytics": ["data", "analy", "dashboard", "report", "insight", "tableau", "power bi"],
    "data engineering": ["pipeline", "etl", "data", "warehouse", "spark", "airflow"],
    "data mining": ["mining", "data"],
    "data visualization": ["visuali", "dashboard", "chart", "plot", "tableau", "power bi", "matplotlib"],
    "big data": ["big data", "spark", "hadoop", "hive", "kafka"],
    "etl": ["etl", "extract", "pipeline", "transform"],
    "data warehousing": ["warehouse", "snowflake", "redshift", "bigquery"],
    "database design": ["database", "schema", "sql", "relation", "normaliz"],
    "query optimization": ["query", "optimi", "index"],
    "time series": ["time series", "time-series", "forecast"],
    "object detection": ["detect", "yolo", "object"],
    "image processing": ["image", "opencv", "photo"],
    "image segmentation": ["segment"],
    "image classification": ["classif", "image"],
    "face recognition": ["face", "facial"],
    "ocr": ["ocr", "optical", "text recogni", "tesseract", "scan"],
    "speech recognition": ["speech", "voice", "audio", "whisper"],
    "recommendation systems": ["recommend"],
    "sentiment analysis": ["sentiment", "opinion"],
    "text classification": ["text", "classif"],
    "rest api": ["api", "rest", "endpoint", "http", "backend", "back-end", "integrat", "fastapi", "flask",
                 "express", "django", "graphql", "postman", "swagger"],
    "api development": ["api", "rest", "endpoint", "http", "backend", "back-end", "fastapi", "flask", "express",
                        "django", "graphql"],
    "json": ["json", "api", "rest", "serializ"],
    "microservices": ["microservice", "service", "docker", "kubernetes"],
    "soa": ["service"],
    "state management": ["state", "provider", "bloc", "getx", "riverpod", "redux", "mobx", "zustand", "vuex",
                         "context api"],
    "mvc": ["mvc", "mvvm", "mvp", "architect", "model-view", "design pattern", "laravel", "codeigniter",
            "django", "rails", "spring mvc", "asp.net", "symfony", "nestjs"],
    "mvvm": ["mvc", "mvvm", "mvp", "architect", "model-view", "design pattern", "viewmodel", "jetpack",
             "swiftui", "provider", "bloc"],
    "mvp": ["mvc", "mvvm", "mvp", "architect", "model-view"],
    "clean architecture": ["clean", "architect", "layer"],
    "oop": ["object", "oop", "class", "inherit"],
    "tdd": ["test"], "bdd": ["test", "behavio"],
    "agile": ["agile", "scrum", "sprint", "kanban", "jira"], "scrum": ["scrum", "sprint", "agile"],
    "ci/cd": ["ci", "cd", "pipeline", "jenkins", "github action", "gitlab", "deploy"],
    "devops": ["devops", "docker", "kubernetes", "ci", "deploy", "terraform", "ansible"],
    "mlops": ["mlops", "mlflow", "deploy", "model"],
    "version control": ["git", "svn", "version"],
    "full stack": ["full", "stack", "frontend", "front-end", "backend", "back-end"],
    "cloud": ["cloud", "aws", "azure", "gcp", "firebase", "heroku", "vercel"],
    "serverless": ["serverless", "lambda", "function"],
    "embedded systems": ["embedded", "microcontroller", "arduino", "firmware"],
    "iot": ["iot", "sensor", "arduino", "raspberry"],
    "robotics": ["robot", "ros"],
    "sdk": ["sdk", "library", "librar", "integrat", "plugin", "package"],
    "ai agents": ["agent"], "function calling": ["function", "tool"],
    "semantic search": ["semantic", "search", "embedding"], "quantization": ["quantiz"],
    "account management": ["account", "client", "customer"], "key account management": ["account", "client"],
    "sales": ["sales", "sell", "revenue", "deal"], "business development": ["business", "partner", "growth"],
    "lead generation": ["lead"], "negotiation": ["negotiat", "deal", "contract"],
    "customer service": ["customer", "client", "service", "support"],
    "marketing": ["market", "campaign", "brand"], "digital marketing": ["digital", "seo", "social", "campaign"],
    "market research": ["research", "market", "survey"],
    "recruitment": ["recruit", "hiring", "talent", "sourcing"], "onboarding": ["onboard"],
    "employee relations": ["employee"], "performance management": ["performance"],
    "budgeting": ["budget"], "forecasting": ["forecast"], "financial analysis": ["financ", "analy"],
    "supply chain management": ["supply"], "procurement": ["procure", "purchas", "vendor"],
    "inventory management": ["inventory", "stock"], "logistics": ["logistic", "shipping", "transport"],
    "project management": ["project", "manag", "plan", "coordinat"], "event planning": ["event"],
    "public relations": ["pr", "media", "public"], "content writing": ["content", "writ", "blog"],
    "copywriting": ["copy", "writ"],
    "compensation and benefits": ["compensation", "benefit", "salar", "payroll", "reward"],
    "labour law": ["labour", "labor", "law", "legal", "compliance", "eobi", "sessi"],
    "employer branding": ["brand", "career fair", "campus"], "employee engagement": ["engag", "morale"],
    "training and development": ["training", "learning", "develop", "workshop"],
    "succession planning": ["succession"], "talent management": ["talent"],
    "hr operations": ["hr", "human resource", "employee record", "attendance"],
    "hr policies": ["polic", "handbook"],
    "mobile development": ["mobile", "android", "ios", "flutter", "app"],
    "cross-platform development": ["cross", "platform", "flutter", "react native"],
    "unit testing": ["unit", "test"], "test automation": ["automat", "test"], "software testing": ["test", "qa"],
    "infrastructure as code": ["terraform", "infrastructure", "cloudformation", "pulumi", "ansible"],
    "containerization": ["container", "docker", "kubernetes"],
    "orm": ["orm", "hibernate", "sqlalchemy", "prisma", "sequelize", "eloquent", "entity"],
    "data pipelines": ["pipeline", "etl", "airflow", "ingest"],
    "business intelligence": ["bi", "dashboard", "report", "power bi", "tableau", "looker"],
    "message queues": ["queue", "kafka", "rabbitmq", "broker", "pub"],
    "event-driven architecture": ["event", "kafka", "queue"],
    "ui/ux": ["ui", "ux", "design", "figma", "wirefram", "prototyp"],
    "financial reporting": ["financ", "report", "statement", "balance"],
    "bookkeeping": ["book", "ledger", "journal", "reconcil"],
}


def evidence_supports(skills, line: str) -> bool:
    """True if `line` mentions something related to any of `skills`."""
    low = (line or "").lower()
    for skill in skills:
        stems = list(CONCEPT_HINTS.get(skill, []))
        stems += [w for w in re.findall(r"[a-z0-9]+", skill) if len(w) >= 3]
        stems += implying_sources(skill)
        for st in stems:
            pat = re.escape(st) if len(st) >= 4 else r"(?<![a-z0-9])" + re.escape(st) + r"(?![a-z])"
            if re.search(pat, low):
                return True
    return False


_SQL_DBS = ["mysql", "postgresql", "sql server", "mariadb", "sqlite", "oracle", "sql"]
RELATED = {db: [x for x in _SQL_DBS if x != db] for db in _SQL_DBS if db != "sql"}
RELATED.update({
    "django": ["flask", "fastapi"], "flask": ["django", "fastapi"], "fastapi": ["flask", "django"],
    "pytorch": ["tensorflow", "keras"], "tensorflow": ["pytorch", "keras"],
    "aws": ["azure", "google cloud"], "azure": ["aws", "google cloud"], "google cloud": ["aws", "azure"],
    "mongodb": ["firebase", "dynamodb", "couchbase", "cassandra"],
})


COMPOSITE_STACKS = {
    "rag": [sorted(RAG_RETRIEVAL_SKILLS), sorted(RAG_LLM_SKILLS)],
    "full stack": [["react", "vue", "angular", "next.js", "nuxt.js", "svelte", "html"],
                   ["nodejs", "django", "flask", "fastapi", "spring boot", "laravel", "express", "php", "nestjs"],
                   ["sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "oracle", "sql server", "nosql",
                    "firebase", "supabase"]],
    "mern": [["mongodb"], ["express"], ["react", "react native"], ["nodejs"]],
    "mean": [["mongodb"], ["express"], ["angular"], ["nodejs"]],
    "lamp": [["linux"], ["apache"], ["mysql", "mariadb"], ["php", "python", "perl"]],
}
COMPOSITE_KIND = {"rag": MODERATE}
_STACK_WINDOW = 300

SKILL_TEXT_PATTERNS = {
    "artificial intelligence": [r"(?i:\bartificial\s+intelligence\b)|\bAI\b"],
    "machine learning": [r"(?i:\bmachine[\s-]+learning\b)|\bML\b"],
    "deep learning": [r"(?i:\bdeep[\s-]+learning\b)"],
    "natural language processing": [r"(?i:\bnatural\s+language\s+processing\b|\bnlp\b)"],
    "computer vision": [r"(?i:\bcomputer\s+vision\b)"],
    "rest api": [r"(?i:\b(?:rest(?:ful)?\s*(?:apis?|services?|endpoints?)|" + _NOT_REST_LOOKBEHIND + r"apis?)\b)"],
    "ci/cd": [r"(?i:\bci\s*/?\s*cd\b|\bcontinuous\s+integration\b)"],
    "git": [r"(?i:\bgit(?:hub|lab)?\b)"],
}


FOUND, INFERRED, NOT_FOUND = "found", "inferred", "not found"


@dataclass
class SkillMatch:
    skill: str
    matched: bool
    match_type: str = "none"
    via: str = ""
    evidence: str = ""
    quality: float = 0.0
    confidence: str = ""
    basis: str = ""
    status: str = ""
    relation: str = ""

    @property
    def hands_on(self) -> bool:
        """The CV shows the skill itself USED in work or a project — named, or
        as an instance of it (FAISS for "vector database")."""
        return self.matched and (self.match_type == "direct" or self.relation == ISA)             and self.basis == _BASIS["applied"]

    def __post_init__(self):
        if not self.status:
            self.status = (NOT_FOUND if not self.matched
                           else FOUND if self.match_type in ("direct", "course") else INFERRED)
        if self.status == INFERRED and not self.confidence:
            self.confidence = {"llm": MODERATE, "related": WEAK}.get(self.match_type, _confidence(self.quality))


def _snippet(text: str, start: int, end: int, width: int = 70) -> str:
    """The CV line around [start, end), trimmed to ~2*width chars."""
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    le = len(text) if le == -1 else le
    s = max(ls, start - width)
    e = min(le, end + width)
    snip = text[s:e].strip(" •·-–—|\t")
    return ("…" if s > ls else "") + snip + ("…" if e < le else "")


_WEAK_LINE_RE = re.compile(
    r"\b(?:courses?|certificat\w*|certified|fundamentals|beginners?|bootcamp|udemy|coursera|edx|"
    r"tutorials?|workshops?|crash\s+course|nanodegree|online\s+class|training\s+(?:program|course|session))\b"
    r"|\bcurrently\s+learning\b|\beager\s+to\s+learn\b|\blearning\s+to\b|:\s*learn(?:ing)?\b|^\W*learn(?:ing)?\b",
    re.IGNORECASE)

_APPLIED_LINE_RE = re.compile(
    r"\b(?:built|building|develop(?:ed|ing)|implement(?:ed|ing)|design(?:ed|ing)|creat(?:ed|ing)|trained|"
    r"fine[\s-]?tuned|deploy(?:ed|ing)|integrat(?:ed|ing)|engineered|architected|optimi[sz](?:ed|ing)|"
    r"automat(?:ed|ing)|delivered|launched|shipped|led|used|using|leverag(?:ed|ing)|utili[sz](?:ed|ing)|"
    r"powered\s+by|worked\s+(?:on|with)|working\s+on|responsible\s+for)\b",
    re.IGNORECASE)

_CAPS_HEADING_RE = re.compile(r'[^\w\n]*(?:[A-Z]{2,}[ &/]+)*[A-Z]{4,}(?=\s*(?:[•:\-–|]|\n|$))')

_BASIS = {"applied": "used in work / projects", "listed": "listed only", "course": "course / learning only"}


class CVIndex:
    """Pre-computed view of one CV for fast repeated matching."""

    def __init__(self, text: str, cv_skills: set):
        self.text = text or ""
        self.skills = set(cv_skills or ())
        self._positions = {}
        self._all = {}
        for canon, s, e in skill_mentions(self.text):
            self._positions.setdefault(canon, (s, e))
            self._all.setdefault(canon, []).append((s, e))
        from modules.experience import _find_section_spans
        spans = _find_section_spans(self.text)
        self._applied_spans = [(a, b) for name, a, b in spans if name in ("experience", "projects")]
        self._course_spans = []
        for name, a, b in spans:
            if name == "certifications":
                cut = a
                for _ in range(15):
                    nxt = self.text.find("\n", cut + 1)
                    if nxt == -1 or nxt >= b:
                        cut = b
                        break
                    if _CAPS_HEADING_RE.match(self.text, nxt + 1):
                        cut = nxt
                        break
                    cut = nxt
                self._course_spans.append((a, min(b, cut)))
        self._strength = {}
        self._usage = {}

    def has(self, skill: str) -> bool:
        return skill in self.skills

    def _mentions(self, skill: str) -> list:
        spans = list(self._all.get(skill, []))
        for pat in SKILL_TEXT_PATTERNS.get(skill, []):
            spans += [(m.start(), m.end()) for m in re.finditer(pat, self.text)]
        if not spans:
            spans = [(m.start(), m.end()) for m in re.finditer(
                r'(?<![A-Za-z\d])' + re.escape(skill) + r'(?![A-Za-z\d])', self.text, re.IGNORECASE)]
        return sorted(set(spans))

    def _line(self, pos: int) -> str:
        ls = self.text.rfind("\n", 0, pos) + 1
        le = self.text.find("\n", pos)
        return self.text[ls:le if le != -1 else len(self.text)]

    def _is_weak(self, start: int) -> bool:
        if any(a <= start < b for a, b in self._course_spans):
            return True
        return bool(_WEAK_LINE_RE.search(self._line(start)))

    def is_applied(self, start: int) -> bool:
        """The mention at `start` is in a job / project, or on a line describing work done."""
        if self._is_weak(start):
            return False
        return any(a <= start < b for a, b in self._applied_spans) or bool(_APPLIED_LINE_RE.search(self._line(start)))

    def strength(self, skill: str) -> str:
        """"strong" if any mention is real use, "weak" if every mention is a
        course / certificate / learning line, "strong" when no position is
        known (skill came from token discovery)."""
        if skill not in self._strength:
            spans = self._mentions(skill)
            weak = bool(spans) and all(self._is_weak(sp[0]) for sp in spans)
            self._strength[skill] = "weak" if weak else "strong"
        return self._strength[skill]

    def usage(self, skill: str) -> str:
        """How the CV shows `skill`: "applied" (in a job/project or a line describing work),
        "listed" (only in a skills list / keyword line) or "course" (only studied)."""
        if skill not in self._usage:
            spans = self._mentions(skill)
            if self.strength(skill) == "weak":
                self._usage[skill] = "course"
            elif any(self.is_applied(s) for s, _ in spans):
                self._usage[skill] = "applied"
            else:
                self._usage[skill] = "listed"
        return self._usage[skill]

    def evidence_for(self, skill: str) -> str:
        """A CV quote for `skill`, preferring a line that shows it used."""
        spans = self._mentions(skill)
        strong = [sp for sp in spans if not self._is_weak(sp[0])]
        applied = [sp for sp in strong if self.is_applied(sp[0])]
        sp = (applied or strong or spans or [None])[0]
        return _snippet(self.text, *sp) if sp else ""

    def search(self, pattern: str):
        return re.search(pattern, self.text, re.IGNORECASE)


def _direct(skill: str, cv: CVIndex) -> bool:
    if skill in COMPOSITE_STACKS:
        return skill in cv._positions or any(re.search(p, cv.text) for p in SKILL_TEXT_PATTERNS.get(skill, []))
    if cv.has(skill):
        return True
    for pat in SKILL_TEXT_PATTERNS.get(skill, []):
        if re.search(pat, cv.text):
            return True
    return False


def _weak_credit(kind: str, cv: CVIndex, via_skill: str) -> float:
    return min(MATCH_CREDIT[kind], MATCH_CREDIT["course"]) if cv.strength(via_skill) == "weak" else MATCH_CREDIT[kind]


_LEVELS = (STRONG, MODERATE, WEAK)
_EPS = 1e-9


def _level(kind: str, usage: str):
    """Confidence of one implication given how the implying skill appears in the CV.
    ISA holds even for a listed skill; anything else needs the skill shown in use
    to keep its strength, and drops a level (or out) when it is only listed."""
    if kind == ISA:
        return STRONG
    if usage == "applied":
        return kind
    i = _LEVELS.index(kind) + 1
    return _LEVELS[i] if i < len(_LEVELS) else None


def _confidence(credit: float) -> str:
    """The level whose credit is nearest (a two-step chain earns e.g. 0.9 × 0.75)."""
    return min(_LEVELS, key=lambda lv: (abs(INFERENCE_CREDIT[lv] - credit), _LEVELS.index(lv)))


def _inferred(skill: str, kind: str, usage: str, via: str, evidence: str):
    level = _level(kind, usage)
    if level is None:
        return None
    credit = INFERENCE_CREDIT[level]
    if usage == "course":
        return SkillMatch(skill, True, "course", f"{via} (course / learning only)", evidence,
                          min(credit, MATCH_CREDIT["course"]), WEAK, _BASIS["course"], status=INFERRED,
                          relation=kind)
    return SkillMatch(skill, True, "inferred", via, evidence, credit, level, _BASIS[usage], relation=kind)


def _stack(skill: str, cv: CVIndex):
    """A composite stack (MERN, full stack, RAG) shown by one skill from every group. It counts as
    "used" only when the groups appear TOGETHER in work (same role / project), not scattered."""
    groups = [[g for g in group if _direct(g, cv)] for group in COMPOSITE_STACKS[skill]]
    if not all(groups):
        return None
    parts = [g[0] for g in groups]
    if any(all(cv.usage(h) == "course" for h in g) for g in groups):
        usage = "course"
    else:
        usage = "listed"
        others = [[s for h in g for s, _ in cv._mentions(h)] for g in groups[1:]]
        for pivot in groups[0]:
            for s, _ in cv._mentions(pivot):
                if cv.is_applied(s) and all(any(abs(o - s) <= _STACK_WINDOW for o in grp) for grp in others):
                    usage, parts = "applied", [pivot] + [next(h for h in g for o, _ in cv._mentions(h)
                                                              if abs(o - s) <= _STACK_WINDOW) for g in groups[1:]]
                    break
            if usage == "applied":
                break
    return _inferred(skill, COMPOSITE_KIND.get(skill, ISA), usage, " + ".join(parts), cv.evidence_for(parts[0]))


def _infer(skill: str, cv: CVIndex, depth: int = 0):
    """The best-supported inference for `skill`, or None. Follows at most two
    implications (AI ⇐ machine learning ⇐ scikit-learn); each step multiplies the
    credit, so a chain is never stronger than its weakest link, and anything below
    a weak inference is dropped (no over-inference through long chains)."""
    best = None

    def consider(m):
        nonlocal best
        if m and m.quality >= INFERENCE_CREDIT[WEAK] - _EPS and (best is None or m.quality > best.quality + _EPS):
            best = m

    if skill in COMPOSITE_STACKS:
        consider(_stack(skill, cv))
    for src, kind in IMPLIED_BY.get(skill, []):
        if src.startswith("re:"):
            m = cv.search(src[3:])
            if m:
                usage = "course" if cv._is_weak(m.start()) else "applied" if cv.is_applied(m.start()) else "listed"
                said = re.sub(r"\s+", " ", m.group(0)).strip()
                via = "“" + (said if len(said) <= 40 else said[:37] + "…") + "”"
                consider(_inferred(skill, kind, usage, via, _snippet(cv.text, m.start(), m.end())))
        elif _direct(src, cv):
            consider(_inferred(skill, kind, cv.usage(src), src, cv.evidence_for(src)))
    if depth == 0 and skill in CONCEPT_SKILLS:
        for src, kind in IMPLIED_BY.get(skill, []):
            if src.startswith("re:") or (src not in IMPLIED_BY and src not in COMPOSITE_STACKS) or _direct(src, cv):
                continue
            sub = _infer(src, cv, depth + 1)
            if sub:
                q = round(sub.quality * (1.0 if kind == ISA else INFERENCE_CREDIT[kind]), 3)
                consider(SkillMatch(skill, True, sub.match_type, f"{src} ← {sub.via}", sub.evidence, q,
                                    WEAK if sub.match_type == "course" else _confidence(q), sub.basis,
                                    status=INFERRED, relation=ISA if kind == ISA == sub.relation else kind))
    return best


def match_skill(skill: str, cv: CVIndex, _depth: int = 0) -> SkillMatch:
    """Found (named in the CV, any alias) → Inferred (implied by evidence in the CV,
    with a confidence) → Inferred weak (a sibling skill) → Not found."""
    skill = canonicalize_skill(skill)

    if _direct(skill, cv):
        usage = cv.usage(skill)
        if usage == "course":
            return SkillMatch(skill, True, "course", "course / learning only", cv.evidence_for(skill),
                              MATCH_CREDIT["course"], basis=_BASIS["course"])
        return SkillMatch(skill, True, "direct", "", cv.evidence_for(skill), MATCH_CREDIT["direct"],
                          basis=_BASIS[usage])

    inferred = _infer(skill, cv)
    if inferred:
        return inferred

    for sibling in RELATED.get(skill, []):
        if _direct(sibling, cv):
            return SkillMatch(skill, True, "related", sibling, cv.evidence_for(sibling),
                              _weak_credit("related", cv, sibling), WEAK, _BASIS[cv.usage(sibling)],
                              relation="related")

    return SkillMatch(skill, False)


def match_requirements(jd_skills, cv: CVIndex) -> dict:
    """{canonical_skill: SkillMatch} for every JD skill."""
    return {canonicalize_skill(s): match_skill(s, cv) for s in jd_skills}


