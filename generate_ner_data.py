import random
import re
import pandas as pd



manufacturing_brands = [
    "DeWalt", "Milwaukee Tool", "Snap-on", "Proto", "Ridgid", "Jet Tools", "Powermatic", "Greenlee",
    "Crescent Tools", "Irwin Tools", "Klein Tools", "Makita", "Hilti", "Lufkin", "Bosch Tools",
    "Southwire", "Gardner Bender", "Ideal Industries", "Fluke", "Klein", "UEi Test Instruments",
    "Yellow Jacket", "Fieldpiece", "Testo", "AEMC Instruments", "Allen-Bradley", "FactoryTalk",
    "Square D", "Siemens LOGO!", "Eaton Cutler-Hammer", "Phoenix Contact", "Omron", "ABB Ability",
    "Schneider Modicon", "Banner Engineering", "3M", "Honeywell Safety", "Ansell", "Brady", "MSA Safety",
    "Tyvek", "Tingley", "Wolverine Procore", "Ergodyne", "Jackson Safety", "Grainger", "Zoro", "Fastenal",
    "MSC Industrial", "Motion Industries"
]

tech_brands = [
    "AWS", "Amazon EC2", "Amazon S3", "Azure", "Microsoft Power BI", "Google Cloud", "BigQuery", "Cloud Run", "Firebase",
    "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Trello", "CircleCI", "Travis CI", "Docker", "Kubernetes",
    "Salesforce", "HubSpot", "Marketo", "Pardot", "Intercom", "Zendesk", "Freshdesk", "ServiceNow", "Workday", "NetSuite",
    "Databricks", "Snowflake", "Tableau", "Looker", "Power BI", "Alteryx", "Domo", "Mode Analytics", "Apache Superset", "Airbyte",
    "Okta", "Duo Security", "Auth0", "Cloudflare", "CrowdStrike", "SentinelOne", "Zscaler", "1Password", "Tanium", "CyberArk"
]

consulting_brands = [
    "McKinsey", "Deloitte", "Accenture", "EY", "KPMG", "PwC", "Capgemini", "BCG", "Bain & Company"
]


templates = [
    "{brand} announced a partnership with {partner} to accelerate digital transformation.",
    "{brand} expanded its footprint in the enterprise automation space.",
    "The latest release from {brand} introduces AI-powered capabilities for compliance teams.",
    "{brand} was named a leader in the latest analyst report for cloud security.",
    "Organizations across finance are adopting {brand} to streamline operations.",
    "{brand} unveiled its new enterprise-grade data integration engine.",
    "{brand} integrates with {partner} to deliver unified observability.",
    "{brand} has been deployed by Fortune 500 firms to modernize legacy systems.",
    "The deployment of {brand} led to a 25% reduction in operational overhead.",
    "{brand}'s compliance toolkit now supports SOC 2, HIPAA, and GDPR.",
    "{brand} and {partner} collaborate to deliver zero-trust security architecture.",
    "Enterprise teams use {brand} to orchestrate multi-cloud infrastructure.",
    "{brand} completed its rollout across North American manufacturing sites.",
    "The analytics engine in {brand} now includes support for real-time anomaly detection.",
    "Retail chains have standardized on {brand} for inventory optimization.",
    "{brand} is seeing rapid adoption among mid-market SaaS companies.",
    "{brand}'s product roadmap includes deeper integration with {partner}'s APIs.",
    "The migration to {brand} helped reduce licensing costs by 30%.",
    "Support for Kubernetes-native policies is now GA in {brand}.",
    "{brand} delivers consistent security policies across hybrid environments.",
    "The new release of {brand} introduces single sign-on and MFA support.",
    "Early adopters of {brand} report faster onboarding and configuration times.",
    "{brand} is working with {partner} to bring GenAI capabilities to the edge.",
    "{brand} has become the go-to platform for enterprise workload scheduling.",
    "{brand}'s cloud-native features now cover configuration drift detection.",
    "The executive team at {brand} shared plans for international expansion.",
    "According to Gartner, {brand} is one of the fastest-growing platforms in its category.",
    "{brand} introduced pre-built connectors for SAP, Oracle, and Salesforce.",
    "Feedback from early access users highlights the ease of use of {brand}.",
    "{brand} is now ISO 27001 certified, further strengthening its compliance posture.",
    "{brand} and {partner} are co-developing a new data governance framework.",
    "Operational data from {brand} is now directly available in Snowflake.",
    "{brand} supports audit logging and granular RBAC out-of-the-box.",
    "{brand} launched its self-service deployment wizard for enterprise customers.",
    "{brand} now supports on-prem, cloud, and air-gapped deployments.",
    "The deployment of {brand} resulted in improved SLA adherence for IT teams.",
    "{brand} is helping manufacturers transition to smart factory infrastructure.",
    "{brand}'s ML model performance dashboard now offers drift detection.",
    "CIOs are turning to {brand} to centralize policy enforcement.",
    "{brand}'s latest round of updates reduces provisioning time by 40%.",
    "The new API tier from {brand} includes rate limiting and usage analytics.",
    "CS teams rely on {brand} to unify ticketing, alerts, and escalation workflows.",
    "{brand} published a whitepaper on best practices for large-scale container orchestration.",
    "{brand}'s platform now includes policy as code and audit trails.",
    "{brand} supports ingestion of unstructured data from over 200 sources.",
    "Financial institutions trust {brand} for secure document workflow automation.",
    "With {brand}, IT teams now get end-to-end visibility into infrastructure performance.",
    "The {brand} console now allows live debugging across multiple environments.",
    "A recent case study shows {brand} enabled a 45% reduction in cloud spend.",
    "{brand} delivers continuous compliance checks and remediation alerts.",
    "The observability stack from {brand} now includes native OpenTelemetry support."
]




def simple_tokenize(text):
    return re.findall(r'\w+|\S', text)

def generate_labeled_example(brand, template):

    if "{partner}" in template:
        partner = random.choice(all_brands)  
        while partner == brand:
            partner = random.choice(all_brands)
        sentence = template.format(brand=brand, partner=partner)
    else:
        sentence = template.format(brand=brand)

    tokens = re.findall(r'\w+|\S', sentence)
    brand_tokens = re.findall(r'\w+|\S', brand)
    labels = ['O'] * len(tokens)

  
    for i in range(len(tokens) - len(brand_tokens) + 1):
        if tokens[i:i+len(brand_tokens)] == brand_tokens:
            labels[i] = "B-BRAND"
            for j in range(1, len(brand_tokens)):
                labels[i+j] = "I-BRAND"
            break

    return sentence, list(zip(tokens, labels))


def generate_dataset(brands, count=100):
    samples = []
    for _ in range(count):
        brand = random.choice(brands)
        template = random.choice(templates)
        sentence, token_labels = generate_labeled_example(brand, template)
        samples.append({
            "sentence": sentence,
            "tokens_labels": token_labels
        })
    return samples



if __name__ == "__main__":
    all_brands = manufacturing_brands + tech_brands + consulting_brands
    data = generate_dataset(all_brands, count=300) 

    df = pd.DataFrame(data)
    df.to_csv("b2b_ner_dataset.csv", index=False)

    with open("b2b_ner_dataset.conll", "w") as f:
        for item in data:
            for token, label in item["tokens_labels"]:
                f.write(f"{token}\t{label}\n")
            f.write("\n")

    print("Dataset saved as 'b2b_ner_dataset.csv' and 'b2b_ner_dataset.conll'")
