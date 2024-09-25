import os
import requests
from dotenv import load_dotenv
import dspy
from llamaparse import LlamaParser
from jambaai import JambaAI

# Load environment variables
load_dotenv()

# Function to download the Deloitte report
def download_report(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Function to extract information using Dspy
def extract_with_dspy(document_text):
    dspy.settings.configure(lm='openai')

    class ExtractInformation(dspy.Signature):
        """Extract specific information from the Deloitte US 2023 Audit Quality Report."""
        document = dspy.InputField()
        insights = dspy.OutputField(desc="Key insights")
        findings = dspy.OutputField(desc="Key findings")
        conclusions = dspy.OutputField(desc="Key conclusions")

    extract_module = dspy.Module(ExtractInformation)
    result = extract_module(document=document_text)

    return {
        'insights': result.insights,
        'findings': result.findings,
        'conclusions': result.conclusions
    }

# Function to extract information using LlamaParse
def extract_with_llamaparse(filename):
    parser = LlamaParser()
    parsed_document = parser.parse(filename)

    return {
        'table_of_contents': parser.extract_table_of_contents(parsed_document),
        'kpis': parser.extract_key_performance_indicators(parsed_document),
        'recommendations': parser.extract_key_recommendations(parsed_document)
    }

# Function to extract information using JambaAI
def extract_with_jambaai(filename):
    jamba = JambaAI(api_key=os.getenv('JAMBAAI_API_KEY'))
    document = jamba.load_document(filename)

    return {
        'challenges': jamba.extract_key_challenges(document),
        'opportunities': jamba.extract_key_opportunities(document),
        'risks': jamba.extract_key_risks(document)
    }

# Function to combine and process results
def process_results(dspy_results, llamaparse_results, jambaai_results):
    combined_results = {**dspy_results, **llamaparse_results, **jambaai_results}
    
    # Additional processing can be done here, such as removing duplicates,
    # formatting, or further analysis of the extracted information

    return combined_results

# Main function to run the extraction process
def main():
    report_url = "https://www2.deloitte.com/content/dam/Deloitte/us/Documents/audit/us-audit-quality-report-2023.pdf"
    report_filename = "deloitte_audit_quality_report_2023.pdf"

    # Download the report
    download_report(report_url, report_filename)

    # Extract information using each framework
    with open(report_filename, 'r') as f:
        document_text = f.read()
    
    dspy_results = extract_with_dspy(document_text)
    llamaparse_results = extract_with_llamaparse(report_filename)
    jambaai_results = extract_with_jambaai(report_filename)

    # Combine and process results
    final_results = process_results(dspy_results, llamaparse_results, jambaai_results)

    # Output the results
    print(json.dumps(final_results, indent=2))

if __name__ == "__main__":
    main()