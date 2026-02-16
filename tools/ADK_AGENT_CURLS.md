# Logistics Agent API - Complete Workflow (CURL)

This guide provides the complete sequence of `curl` commands to test the end-to-end workflow of the Logistics Agent, from initialization through SAP Freight Agreement, Rate Table Validity creation, and batch upload of rates.

**Base URL**: `http://localhost:8001/invoke`  
**Note**: Replace `YOUR_SESSION_ID` with a unique string for each test session (e.g., `test_session_123`).  
**Note**: Replace file paths with your actual file locations.

---

## Complete Workflow Steps

### Step 1: Initialize Session & Greet
Start the conversation to get the welcome message.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Hello, I want to process a new agreement."'
```

**Expected Response**: Agent greets you and asks for the agreement document.

---

### Step 2: Upload Agreement PDF
Once the agent asks for the agreement, upload the PDF file. The agent will automatically call `extract_agreement_info` tool.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Here is the agreement file."' \
  -F 'file=@/path/to/your/agreement.pdf'
```

**Example with sample file**:
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="test_session_123"' \
  -F 'message="Here is the agreement file."' \
  -F 'file=@/Users/tanushkagoyal/Downloads/NAVIT/LONG TERM TRANSPORTATION SERVICES AGREEMENT_TEST_SAMPLE.pdf'
```

**Expected Response**: Agent extracts agreement details (description, start date, end date, client, carrier, etc.) and shows them for verification.

---

### Step 3: Verify Agreement Details
Confirm the extracted information. The agent will then proceed to create the Freight Agreement in SAP.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="The extracted information is correct. Please create the freight agreement."'
```

**Expected Response**: Agent calls `create_freight_agreement` tool automatically and shows:
- Agreement ID
- Agreement UUID (⚠️ **IMPORTANT**: Save this UUID!)
- Description, dates, currency

---

### Step 4: Acknowledge Freight Agreement Creation
Confirm that the freight agreement was created successfully. The agent will then ask for the rate card.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Great! The freight agreement is created. What is next?"'
```

**Expected Response**: Agent asks for the rate card Excel file.

---

### Step 5: Upload Rate Card Excel
When the agent asks for the rate card, upload the Excel file. The agent will automatically call `select_rate_card_fields` tool.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Uploading the rate card now."' \
  -F 'file=@/path/to/your/rate_card.xls'
```

**Example with sample file**:
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="test_session_123"' \
  -F 'message="Uploading the DHL rate card now."' \
  -F 'file=@/Users/tanushkagoyal/Downloads/NAVIT/DHL Europe Road.xls'
```

**Expected Response**: Agent shows matched columns and field mappings for verification.

---

### Step 6: Verify Rate Card Field Mappings
Confirm the matched columns. The agent will then proceed to create the Rate Table Validity in SAP.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="The field mappings look correct. Please create the rate table validity."'
```

**Expected Response**: Agent calls `create_rate_table_validity` tool automatically (using the Agreement UUID from Step 3) and shows:
- TranspRateTableValidityUUID (⚠️ **IMPORTANT**: Save this UUID for batch upload!)
- Validity dates
- Timezone

---

### Step 7: Batch Upload Rates to SAP
After Rate Table Validity is created, trigger batch upload of rates from the Excel file. The Validity UUID is auto-retrieved from session.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Please batch upload the rates from the rate card to SAP."'
```

**Expected Response**: Agent calls `batch_upload_to_sap` tool automatically (using Validity UUID from Step 6 and Excel file from Step 5) and shows:
- Number of rate entries uploaded
- Validity UUID used
- Response status

---

### Step 8: Complete Workflow
Acknowledge completion. The workflow is now complete with Freight Agreement, Rate Table Validity, and rates uploaded to SAP.
```bash
curl -X POST http://localhost:8001/invoke \
  -F 'session_id="YOUR_SESSION_ID"' \
  -F 'message="Perfect! The workflow is complete."'
```

**Expected Response**: Agent confirms completion and summarizes what was created.

---

## Complete Example Script

Here's a complete bash script you can run (update file paths and session ID):

```bash
#!/bin/bash

SESSION_ID="test_session_$(date +%s)"
BASE_URL="http://localhost:8001/invoke"
AGREEMENT_FILE="/path/to/your/agreement.pdf"
RATE_CARD_FILE="/path/to/your/rate_card.xls"

echo "=== Step 1: Initialize Session ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Hello, I want to process a new agreement."'

echo -e "\n\n=== Step 2: Upload Agreement PDF ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Here is the agreement file."' \
  -F "file=@$AGREEMENT_FILE"

echo -e "\n\n=== Step 3: Verify Agreement Details ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="The extracted information is correct. Please create the freight agreement."'

echo -e "\n\n=== Step 4: Acknowledge Freight Agreement ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Great! The freight agreement is created. What is next?"'

echo -e "\n\n=== Step 5: Upload Rate Card ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Uploading the rate card now."' \
  -F "file=@$RATE_CARD_FILE"

echo -e "\n\n=== Step 6: Verify Rate Card Mappings ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="The field mappings look correct. Please create the rate table validity."'

echo -e "\n\n=== Step 7: Batch Upload Rates ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Please batch upload the rates from the rate card to SAP."'

echo -e "\n\n=== Step 8: Complete ==="
curl -X POST $BASE_URL \
  -F "session_id=$SESSION_ID" \
  -F 'message="Perfect! The workflow is complete."'

echo -e "\n\n=== Workflow Complete! ==="
```

---

## Important Notes

1. **Session ID**: Keep the same `session_id` throughout the entire workflow to maintain conversation context.

2. **Agreement UUID**: After Step 3, the agent will return an Agreement UUID. The agent automatically uses this UUID in Step 6 when creating Rate Table Validity.

3. **Rate Table Validity UUID**: After Step 6, the agent stores `TranspRateTableValidityUUID` in session - it's auto-retrieved for batch upload in Step 7.

4. **Tool Calls**: The agent automatically calls tools based on the conversation flow:
   - Step 2 → `extract_agreement_info`
   - Step 3 → `create_freight_agreement` (creates Agreement + Calc Sheet + Rate Validity)
   - Step 5 → `select_rate_card_fields`
   - Step 6 → `create_rate_table_validity`
   - Step 7 → `batch_upload_to_sap`

5. **Error Handling**: If any step fails, check the response for error messages. You may need to retry with corrected information.

---

## Starting the Service

If the service is not running, start it from the root directory:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 main.py
```

Or with a virtual environment:

```bash
export PYTHONPATH=$PYTHONPATH:.
source /path/to/venv/bin/activate
python3 main.py
```

The service will start on `http://localhost:8001` (as configured in `config/dev_config.py`).

