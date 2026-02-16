import json
import logging
import fitz  # PyMuPDF
import requests
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
from google.adk.tools import FunctionTool
from config.dev_config import (
    AZURE_OPENAI_CHAT_URL,
    AZURE_OPENAI_API_KEY,
)

# Import Phoenix tracing
from utils.phoenix_tracing import spa_upload_tracer

logger = logging.getLogger(__name__)

def calculate_end_date(start_date_str, no_of_years_str):
    if not start_date_str or not no_of_years_str:
        return None
    try:
        start_date = parse_date(start_date_str)
        num_years = float(no_of_years_str)
        end_date = start_date + relativedelta(months=int(num_years * 12))
        return end_date.strftime('%Y-%m-%d')
    except Exception:
        return None

@spa_upload_tracer.tool(name="extract_agreement_info", description="Extracts client, carrier, start_date, and no_of_years from a PDF agreement file path")
def extract_agreement_info(file_path: str) -> str:
    """Extracts client, carrier, start_date, and no_of_years from a PDF agreement file path."""
    logger.info(f"Tool extract_agreement_info called for: {file_path}")

    extracted_data = {"client": None, "carrier": None, "start_date": None, "no_of_years": None}

    try:
        # Basic validation: ensure we are dealing with a PDF, not an Excel or other file.
        if not file_path or not isinstance(file_path, str):
            return "Error extracting info: Invalid file path provided for agreement PDF."

        if not file_path.lower().endswith(".pdf"):
            return (
                "Error extracting info: The agreement extraction tool only supports PDF files. "
                f"Received file: '{file_path}'. Please upload the agreement as a PDF, not an Excel file."
            )

        doc = fitz.open(file_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text() + "\n"
        doc.close()

        prompt = f"""
        You are a data extraction assistant. Analyze the agreement text and extract these fields:
        - client: The name of the client company.
        - carrier: The name of the provider/carrier (e.g. DHL, FedEx).
        - start_date: The agreement start date (YYYY-MM-DD).
        - no_of_years: The duration/term in years (number only).
        
        Agreement Text:
        {all_text[:20000]}
        
        Return ONLY valid JSON with keys: client, carrier, start_date, no_of_years.
        If a field is not present in the text, set its value to "Not found" (string) but do not omit any key.
        """

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        print(payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        }

        logger.info("Calling Azure OpenAI for agreement extraction...")
        response = requests.post(
            AZURE_OPENAI_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=900,
        )
        print(response)
        if response.status_code >= 400:
            logger.error(
                "Azure OpenAI HTTP %s error response (agreement extractor): %s",
                response.status_code,
                response.text,
            )
            response.raise_for_status()

        resp_data = response.json()
        print(resp_data)
        choices = resp_data.get("choices", [])
        if not choices:
            return "Error: Azure OpenAI response missing 'choices'"

        content = choices[0].get("message", {}).get("content", "{}") or "{}"

        # Strip possible markdown code fences like ```json ... ```
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return f"Error: Model returned invalid JSON. Response: {content[:200]}"

        for key in extracted_data:
            if data.get(key):
                extracted_data[key] = data[key]

        if extracted_data["start_date"] and extracted_data["no_of_years"]:
            extracted_data["end_date"] = calculate_end_date(extracted_data["start_date"], extracted_data["no_of_years"])
        else:
            extracted_data["end_date"] = "Not found"

        result = "EXTRACTED AGREEMENT DETAILS:\n"
        result += f"- Client: {extracted_data['client'] or 'Not found'}\n"
        result += f"- Carrier: {extracted_data['carrier'] or 'Not found'}\n"
        result += f"- Start Date: {extracted_data['start_date'] or 'Not found'}\n"
        result += f"- End Date: {extracted_data['end_date']}\n\n"

        return result

    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return f"Error extracting info: {str(e)}"

# Create tool with tracing - the decorator is already on the function
# FunctionTool will call the decorated function, so tracing will work
agreement_tool = FunctionTool(extract_agreement_info)

