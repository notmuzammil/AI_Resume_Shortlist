import pytest

from modules.matcher import CVIndex, match_skill
from modules.skill_extractor import canonicalize_skill, extract_skills


@pytest.mark.parametrize("text,expected", [
    ("Experienced in NLP", "natural language processing"),
    ("natural language processing", "natural language processing"),
    ("Tensor-flow and Scikit learn", "tensorflow"),
    ("Scikit learn", "scikit-learn"),
    ("Lang-chain, Llama Index", "llamaindex"),
    ("Node.js and Express.js", "nodejs"),
    ("Large Language Models (LLMs)", "llm"),
    ("YOLOv5 detection", "yolo"),
    ("HTML5, CSS3", "html"),
    ("Python3", "python"),
    ("C/C++", "c"),
    ("R programming", "r"),
    ("RESTful APIs", "rest api"),
    ("ML engineer", "machine learning"),
])
def test_aliases_and_abbreviations(text, expected):
    assert expected in extract_skills(text)


@pytest.mark.parametrize("text,not_expected", [
    ("I want to express my ideas", "express"),
    ("I excel at teamwork", "excel"),
    ("the rest of the team", "rest api"),
    ("no less than 2 years", "less"),
    ("Spring 2024 semester", "spring"),
    ("each node in the graph", "nodejs"),
    ("I react quickly to change", "react"),
    ("Section C of the report", "c"),
    ("SOFTWARE ENGINEERING EDUCATION", "software"),
    ("AI-powered features", "ai-powered"),
])
def test_no_false_positives_from_english_words(text, not_expected):
    assert not_expected not in extract_skills(text)


def test_nlp_equals_full_form_both_directions():
    assert canonicalize_skill("NLP") == canonicalize_skill("Natural Language Processing")
    cv = CVIndex("Built text classifiers with NLP", extract_skills("Built text classifiers with NLP"))
    assert match_skill("natural language processing", cv).matched
    cv2 = CVIndex("Natural Language Processing course", extract_skills("Natural Language Processing course"))
    assert match_skill("nlp", cv2).matched


@pytest.mark.parametrize("requirement,cv_text,via", [
    ("neural networks", "Implemented CNN-based image classifiers", "cnn"),
    ("nosql", "Data stored in MongoDB", "mongodb"),
    ("natural language processing", "Fine-tuned BERT for sentiment", "bert"),
    ("supervised learning", "Trained classification models on labelled data", "classification models"),
    ("computer vision", "Object detection with YOLOv8 and OpenCV", "yolo"),
    ("version control", "Code on GitHub", "git"),
    ("android", "Apps built with Kotlin", "kotlin"),
])
def test_semantic_inference(requirement, cv_text, via):
    m = match_skill(requirement, CVIndex(cv_text, extract_skills(cv_text)))
    assert m.matched and m.match_type == "inferred"
    assert via in m.via
    assert m.evidence
    assert m.quality < 1.0


def test_composite_stack_inferred():
    text = "MongoDB, Express.js, React, Node.js"
    m = match_skill("mern", CVIndex(text, extract_skills(text)))
    assert m.matched and "mongodb" in m.via


def test_unrelated_cv_matches_nothing():
    text = "Travel agent. Booked hotels and handled reservations for clients."
    cv = CVIndex(text, extract_skills(text))
    for s in ("python", "machine learning", "rest api", "neural networks", "git"):
        assert not match_skill(s, cv).matched


@pytest.mark.parametrize("skills,line,ok", [
    (("state management",), "MediaPipe, and InsightFace in Python.", False),
    (("face recognition",), "MediaPipe, and InsightFace in Python.", True),
    (("supervised learning",), "Trained classifiers on labelled crop images", True),
    (("state management",), "Built the app with Provider and BLoC", True),
    (("mvc", "mvvm", "clean architecture"), "Followed MVVM pattern with repositories", True),
    (("json",), "Hobbies: cricket and travelling", False),
    (("mvc", "mvvm", "clean architecture"), "Full-Stack Developer with expertise in Laravel, CodeIgniter", True),
    (("rest api", "api development"), "Integrated third-party payment endpoints", True),
])
def test_llm_evidence_must_relate_to_requirement(skills, line, ok):
    from modules.matcher import evidence_supports
    assert evidence_supports(skills, line) == ok


def test_github_and_gitlab_count_as_git():
    for text in ("Code on GitHub", "CI on GitLab", "Git, Bitbucket"):
        assert match_skill("git", CVIndex(text, extract_skills(text))).matched


def test_course_only_evidence_gets_partial_credit():
    text = "SKILLS\nHTML, CSS\nCERTIFICATIONS\nPython Fundamentals for Beginners (Coursera)\n"
    m = match_skill("python", CVIndex(text, extract_skills(text)))
    assert m.matched and m.match_type == "course" and m.quality == 0.5
    text2 = text + "EXPERIENCE\nBackend Developer: built services in Python\n"
    m2 = match_skill("python", CVIndex(text2, extract_skills(text2)))
    assert m2.match_type == "direct" and m2.quality == 1.0 and "Backend Developer" in m2.evidence


def test_context_api_is_not_a_rest_api():
    text = "Managed global state using Context API in React"
    assert not match_skill("rest api", CVIndex(text, extract_skills(text))).matched


def test_related_database_partial_credit():
    text = "Databases: MongoDB, SQL"
    m = match_skill("mysql", CVIndex(text, extract_skills(text)))
    assert m.matched and m.match_type == "related" and m.quality == 0.5


def _m(requirement, text):
    return match_skill(requirement, CVIndex(text, extract_skills(text)))


RAG_WORK = ("EXPERIENCE\nAI Engineer, Acme (2022 - 2024)\n"
            "- Built a RAG chatbot over 10k policy documents with LangChain and FAISS\n")


def test_found_inferred_not_found_statuses():
    assert _m("python", "EXPERIENCE\nBuilt APIs in Python").status == "found"
    assert _m("llm", RAG_WORK).status == "inferred"
    assert _m("kubernetes", RAG_WORK).status == "not found"


def test_documented_rag_strongly_implies_llm():
    m = _m("llm", RAG_WORK)
    assert m.status == "inferred" and m.confidence == "strong" and m.quality == 0.9
    assert m.basis == "used in work / projects" and "RAG chatbot" in m.evidence


def test_listed_only_rag_implies_llm_less_strongly():
    m = _m("llm", "SKILLS\nPython, RAG, SQL\n")
    assert m.status == "inferred" and m.confidence == "moderate" and m.quality == 0.75
    assert m.basis == "listed only"


@pytest.mark.parametrize("requirement,text,via", [
    ("deep learning", "PROJECTS\nTrained an image classifier in PyTorch", "pytorch"),
    ("deep learning", "EXPERIENCE\nDeveloped forecasting models using TensorFlow", "tensorflow"),
    ("neural networks", "PROJECTS\nBuilt a helmet detector with YOLOv8", "yolo"),
    ("cnn", "PROJECTS\nBuilt a helmet detector with YOLOv8", "yolo"),
    ("cnn", "PROJECTS\nFine-tuned ResNet-50 for X-ray classification", "ResNet"),
])
def test_strong_evidence_from_frameworks_and_models(requirement, text, via):
    m = _m(requirement, text)
    assert m.status == "inferred" and m.confidence == "strong" and via in m.via


@pytest.mark.parametrize("store", ["FAISS", "ChromaDB", "Chroma", "Pinecone", "Qdrant", "pgvector"])
def test_vector_stores_are_vector_database_evidence_even_when_listed(store):
    text = f"SKILLS\nPython, {store}, Docker\n"
    for requirement in ("vector database", "vector search"):
        m = _m(requirement, text)
        assert m.status == "inferred" and m.confidence == "strong", (requirement, store)


@pytest.mark.parametrize("requirement,text", [
    ("rag", "EXPERIENCE\nBuilt agents with LangChain and the OpenAI API"),
    ("langchain", RAG_WORK.replace(" with LangChain and FAISS", "")),
    ("llamaindex", RAG_WORK),
    ("llm", "PROJECTS\nTransfer learning on a CNN for plant disease detection"),
    ("llm", "SKILLS\nPython, HuggingFace, Pandas\n"),
    ("rest api", "EXPERIENCE\nBuilt a GraphQL gateway in Node.js"),
    ("nosql", "SKILLS\nSupabase, React\n"),
])
def test_no_over_inference(requirement, text):
    m = _m(requirement, text)
    assert not m.matched, m


def test_rag_stack_needs_vector_store_and_llm_in_the_same_work():
    together = _m("rag", "PROJECTS\nDocument Q&A: LangChain pipeline over a FAISS index\n")
    assert together.status == "inferred" and together.confidence == "moderate"
    scattered = _m("rag", "SKILLS\nLangChain, Django, React, Pandas, NumPy, Docker, Kubernetes, AWS, Azure, "
                          "Tableau, Power BI, Excel, Jira, Figma, Photoshop, Illustrator, MongoDB\n"
                          "TOOLS\nFAISS\n")
    assert scattered.status == "inferred" and scattered.confidence == "weak"


def test_inference_chain_decays_and_keeps_its_evidence():
    m = _m("artificial intelligence", "EXPERIENCE\nTrained gradient-boosted models with XGBoost")
    assert m.status == "inferred" and "machine learning" in m.via and "xgboost" in m.via
    assert m.quality < 0.9 and "XGBoost" in m.evidence


def test_inferred_requirement_counts_toward_the_score():
    from modules.pipeline import evaluate_batch
    jd = ("Job Title: LLM Engineer\nRequirements:\n- Experience with LLMs\n- Python\n"
          "- Vector databases\n")
    cv = (b"Jane Doe\nSKILLS\nPython\nEXPERIENCE\nML Engineer, Acme (2021 - 2024)\n"
          b"- Built a RAG assistant over support tickets using LangChain and Pinecone\n")
    c = evaluate_batch(jd, [("jane.txt", cv)]).candidates[0]
    status = {req.label: (q, best) for req, q, best, _ in c.requirement_status}
    llm_q, llm_best = next(v for k, v in status.items() if "LLM" in k.upper())
    assert llm_q == 0.9 and llm_best.status == "inferred"
    assert c.components["skills"].value > 0.9 * 0.9


@pytest.mark.parametrize("requirement,text,via", [
    ("generative ai", RAG_WORK, "rag"),
    ("prompt engineering", RAG_WORK, "rag"),
    ("vector database", "PROJECTS\nBuilt a RAG assistant over HR policies\n", "rag"),
    ("ai agents", "EXPERIENCE\nDeveloped LangChain agents that book meetings\n", "LangChain agents"),
    ("function calling", "PROJECTS\nLLM assistant with tool calling for order lookups\n", "tool calling"),
    ("llm", "PROJECTS\nMulti-agent research assistant built with CrewAI\n", "crewai"),
    ("dart", "EXPERIENCE\nBuilt Flutter apps with Bloc\n", "flutter"),
    ("php", "EXPERIENCE\nBuilt Laravel admin panels\n", "laravel"),
    ("swift", "PROJECTS\nBuilt a SwiftUI habit tracker\n", "swiftui"),
    ("mobile development", "EXPERIENCE\nBuilt React Native apps for logistics\n", "react native"),
    ("cross-platform development", "EXPERIENCE\nBuilt Flutter apps\n", "flutter"),
    ("orm", "EXPERIENCE\nModelled the schema with Prisma and PostgreSQL\n", "prisma"),
    ("data visualization", "EXPERIENCE\nBuilt sales dashboards in Power BI\n", "power bi"),
    ("business intelligence", "EXPERIENCE\nBuilt sales dashboards in Tableau\n", "tableau"),
    ("big data", "EXPERIENCE\nWrote PySpark jobs on Databricks\n", "pyspark"),
    ("data engineering", "EXPERIENCE\nBuilt ELT models in dbt\n", "dbt"),
    ("supervised learning", "EXPERIENCE\nBuilt XGBoost churn models\n", "xgboost"),
    ("kubernetes", "EXPERIENCE\nRan services on EKS\n", "EKS"),
    ("aws", "EXPERIENCE\nStored files in S3 and ran jobs on EC2\n", "S3"),
    ("docker", "EXPERIENCE\nWrote Dockerfiles for every service\n", "Dockerfiles"),
    ("infrastructure as code", "EXPERIENCE\nProvisioned infrastructure with Terraform\n", "terraform"),
    ("unit testing", "EXPERIENCE\nWrote tests with pytest\n", "pytest"),
    ("test automation", "EXPERIENCE\nAutomated UI tests with Selenium\n", "selenium"),
    ("message queues", "EXPERIENCE\nStreamed events through Kafka\n", "kafka"),
    ("bookkeeping", "EXPERIENCE\nMaintained the books in QuickBooks\n", "quickbooks"),
    ("sales", "EXPERIENCE\nClosed enterprise deals worth $2M\n", "Closed enterprise deals"),
])
def test_related_skills_are_inferred(requirement, text, via):
    m = _m(requirement, text)
    assert m.status == "inferred", m
    assert via.lower() in m.via.lower() and m.evidence and m.quality < 1.0


@pytest.mark.parametrize("requirement,text", [
    ("langchain", RAG_WORK.replace(" with LangChain and FAISS", "")),
    ("pytorch", "EXPERIENCE\nTrained deep learning models for defect detection\n"),
    ("huggingface", "PROJECTS\nFine-tuned Llama 3 with LoRA\n"),
    ("docker", "EXPERIENCE\nRan services on EKS\n"),
    ("java", "EXPERIENCE\nBuilt Android apps in Kotlin\n"),
    ("aws", "EXPERIENCE\nUsed a lambda function to sort the list\n"),
    ("sales", "EXPERIENCE\nSalesforce developer: built Apex triggers\n"),
])
def test_related_skills_do_not_over_infer(requirement, text):
    assert not _m(requirement, text).matched


def test_isa_inference_is_hands_on_other_inference_is_knowledge():
    nosql = _m("nosql", "EXPERIENCE\nBuilt the API on MongoDB\n")
    assert nosql.relation == "isa" and nosql.hands_on
    llm = _m("llm", RAG_WORK)
    assert llm.status == "inferred" and llm.relation == "strong" and not llm.hands_on
    listed = _m("python", "SKILLS\nPython, SQL\n")
    assert listed.match_type == "direct" and not listed.hands_on
    assert _m("python", "EXPERIENCE\nBuilt APIs in Python\n").hands_on


@pytest.mark.parametrize("requirement,text,ok", [
    ("recruitment", "EXPERIENCE\nScreened 200+ applicants and scheduled panel interviews\n", True),
    ("recruitment", "PROJECTS\nBuilt an automated CV screening system that ranks candidates\n", False),
    ("ui/ux", "EXPERIENCE\nDesigned wireframes and clickable prototypes in Figma\n", True),
])
def test_business_inferences_need_the_persons_own_work(requirement, text, ok):
    assert _m(requirement, text).matched == ok


def test_xampp_implies_apache_moderately():
    m = _m("apache", "SKILLS\nPHP, MySQL, XAMPP\n")
    assert m.status == "inferred" and m.confidence == "weak"
    assert _m("apache", "EXPERIENCE\nDeployed PHP sites on XAMPP\n").confidence == "moderate"
