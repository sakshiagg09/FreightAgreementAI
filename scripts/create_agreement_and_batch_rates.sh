#!/bin/bash
# End-to-end: Create Freight Agreement -> Get Agreement ID -> GET agreement (expand) ->
#   extract TranspRateTableValidityUUID -> Upload batch rates.
# Requires: curl, jq
# Usage: ./create_agreement_and_batch_rates.sh

set -e
FA_BASE="http://103.152.79.22:8002/sap/opu/odata4/sap/api_transpfreightagreement/srvd_a2x/sap/api_transpfreightagreement/0001"
RT_BASE="http://103.152.79.22:8002/sap/opu/odata4/sap/api_transpratetable/srvd_a2x/sap/transportationratetable/0001"
AUTH="asakshi:Hell@Sakshi122"
COOKIES_FA="cookies_fa.txt"
COOKIES_RT="cookies.txt"
BOUNDARY="batch_123453"

# ----- Step 1: Fetch CSRF for Freight Agreement API -----
echo "Step 1: Fetching CSRF token (Freight Agreement)..."
CSRF_FA=$(curl -s -X GET "${FA_BASE}/" \
  -u "$AUTH" \
  -H "X-CSRF-Token: Fetch" \
  -H "Accept: application/json" \
  -c "$COOKIES_FA" \
  -D - | grep -i "X-CSRF-Token" | cut -d' ' -f2 | tr -d '\r')
if [ -z "$CSRF_FA" ]; then
  echo "Error: Could not get CSRF token for Freight Agreement API."
  exit 1
fi

# ----- Step 2: POST Create Freight Agreement -----
echo "Step 2: Creating Freight Agreement..."
CREATE_BODY='{
  "TransportationAgreementDesc": "Test-Avi 2",
  "TransportationAgreementType": "ZRF1",
  "TranspAgreementValidFrom": "2025-10-01",
  "TranspAgreementValidTo": "2026-09-30",
  "TranspAgreementTimeZone": "CET",
  "TransportationAgreementDocCrcy": "EUR",
  "TransportationShippingType": "",
  "TransportationMode": "",
  "TransportationAgreementStatus": "01",
  "_FreightAgreementItem": [
    {
      "TransportationCalculationSheet": "",
      "_FreightAgrmtCalcSheet": [
        {
          "TransportationCalculationSheet": "",
          "_FrtAgrmtCalcSheetItem": [
            {
              "TranspChargeType": "Z_BASE_FRGHT_RF",
              "TranspCalcResolutionBase": "ROOT",
              "TranspCalcSheetItemCurrency": "EUR",
              "TranspCalcSheetItemAmount": 0.00,
              "TranspChargeInstrnType": "STND",
              "TranspChargeIsDependent": false,
              "TranspRateTableID": "",
              "TransportationStageCategory": "",
              "TranspCalculationMethodName": "",
              "TranspCalcShtItmIsManualCharge": false,
              "_FreightAgreementRateTable": [
                {
                  "TranspRateTableID": "",
                  "TranspRateTableValueType": "A",
                  "TranspChargeType": "Z_BASE_FRGHT_RF",
                  "TranspRateTableSignType": "+",
                  "TranspRateTableTimeZone": "CET",
                  "_FrtAgrmtRateTableScaleRef": [
                    {
                      "TranspRateTableDimensionIndex": "2",
                      "TransportationCalculationBase": "SOURCELOC_ZONE",
                      "TranspRateTblScRefMinValIsSupp": false,
                      "TranspRateTblScRefMaxValIsSupp": false,
                      "TranspRateTblScaleRefScaleType": "X",
                      "TranspRateTblScaleRefQtyUnit": "",
                      "TranspRateTblScaleRefCurrency": "",
                      "TranspRateTblScaleRefCalcType": "A",
                      "TranspRateTblScRefNoValIsSupp": false,
                      "TranspRateScRefIsRlvtForBrkWgt": false
                    },
                    {
                      "TranspRateTableDimensionIndex": "1",
                      "TransportationCalculationBase": "DESTLOC_ZONE",
                      "TranspRateTblScRefMinValIsSupp": false,
                      "TranspRateTblScRefMaxValIsSupp": false,
                      "TranspRateTblScaleRefScaleType": "X",
                      "TranspRateTblScaleRefQtyUnit": "",
                      "TranspRateTblScaleRefCurrency": "",
                      "TranspRateTblScaleRefCalcType": "A",
                      "TranspRateTblScRefNoValIsSupp": false,
                      "TranspRateScRefIsRlvtForBrkWgt": false
                    },
                    {
                      "TranspRateTableDimensionIndex": "3",
                      "TransportationCalculationBase": "GROSS_WEIGHT",
                      "TranspRateTblScRefMinValIsSupp": false,
                      "TranspRateTblScRefMaxValIsSupp": false,
                      "TranspRateTblScaleRefScaleType": "B",
                      "TranspRateTblScaleRefQtyUnit": "KG",
                      "TranspRateTblScaleRefCurrency": "",
                      "TranspRateTblScaleRefCalcType": "A",
                      "TranspRateTblScRefNoValIsSupp": false,
                      "TranspRateScRefIsRlvtForBrkWgt": false
                    }
                  ],
                  "_FrtAgrmtRateTableValidity": []
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "_FrtAgrmtOrganization": [{"FreightAgreementPurchasingOrg": "50000002"}],
  "_FreightAgreementParty": [{"BusinessPartner": "2"}]
}'

CREATE_RESP=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "${FA_BASE}/FreightAgreement" \
  -u "$AUTH" \
  -H "X-CSRF-Token: $CSRF_FA" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -b "$COOKIES_FA" \
  -d "$CREATE_BODY")
HTTP_BODY=$(echo "$CREATE_RESP" | sed '/^HTTP_CODE:/d')
HTTP_CODE=$(echo "$CREATE_RESP" | grep '^HTTP_CODE:' | cut -d: -f2)
if [ "$HTTP_CODE" != "201" ] && [ "$HTTP_CODE" != "200" ]; then
  echo "Error: Create Freight Agreement failed (HTTP $HTTP_CODE)"
  echo "$HTTP_BODY" | jq . 2>/dev/null || echo "$HTTP_BODY"
  exit 1
fi
AGREEMENT_ID=$(echo "$HTTP_BODY" | jq -r '.d.TransportationAgreement // .TransportationAgreement // empty')
AGREEMENT_UUID=$(echo "$HTTP_BODY" | jq -r '.d.TransportationAgreementUUID // .TransportationAgreementUUID // empty')
if [ -z "$AGREEMENT_ID" ] || [ -z "$AGREEMENT_UUID" ]; then
  echo "Error: Could not parse Agreement ID or UUID from response."
  echo "$HTTP_BODY" | jq . 2>/dev/null || echo "$HTTP_BODY"
  exit 1
fi
echo "  Agreement ID: $AGREEMENT_ID"
echo "  Agreement UUID: $AGREEMENT_UUID"

# ----- Step 3: GET Freight Agreement with expand to retrieve TranspRateTableValidityUUID -----
echo "Step 3: Getting Freight Agreement (expand) to retrieve TranspRateTableValidityUUID..."
# Re-fetch CSRF for same FA service (cookie already set)
CSRF_FA2=$(curl -s -X GET "${FA_BASE}/" \
  -u "$AUTH" \
  -H "X-CSRF-Token: Fetch" \
  -H "Accept: application/json" \
  -b "$COOKIES_FA" \
  -c "$COOKIES_FA" \
  -D - | grep -i "X-CSRF-Token" | cut -d' ' -f2 | tr -d '\r')
FILTER="TransportationAgreement%20eq%20%27${AGREEMENT_ID}%27"
EXPAND="_FreightAgreementItem(\$expand=_FreightAgrmtCalcSheet(\$expand=_FrtAgrmtCalcSheetItem(\$expand=_FreightAgreementRateTable(\$expand=_FrtAgrmtRateTableScaleRef,_FrtAgrmtRateTableValidity)))))"
GET_URL="${FA_BASE}/FreightAgreement?\$filter=${FILTER}&\$expand=${EXPAND}"
GET_RESP=$(curl -s -X GET "$GET_URL" \
  -u "$AUTH" \
  -H "X-CSRF-Token: $CSRF_FA2" \
  -H "Accept: application/json" \
  -b "$COOKIES_FA")
# SAP may return .d.results[] or .value[]
VALIDITY_UUID=$(echo "$GET_RESP" | jq -r '
  (.d.results[0] // .value[0] // .d // .) |
  ._FreightAgreementItem[0]._FreightAgrmtCalcSheet[0]._FrtAgrmtCalcSheetItem[0]._FreightAgreementRateTable[0]._FrtAgrmtRateTableValidity[0].TranspRateTableValidityUUID // empty
')
if [ -z "$VALIDITY_UUID" ] || [ "$VALIDITY_UUID" = "null" ]; then
  echo "Error: Could not extract TranspRateTableValidityUUID from GET response."
  echo "Response (first 500 chars): $(echo "$GET_RESP" | head -c 500)"
  exit 1
fi
echo "  TranspRateTableValidityUUID: $VALIDITY_UUID"

# ----- Step 4: Fetch CSRF for Rate Table API -----
echo "Step 4: Fetching CSRF token (Rate Table)..."
CSRF_RT=$(curl -s -X GET "${RT_BASE}/" \
  -u "$AUTH" \
  -H "X-CSRF-Token: Fetch" \
  -H "Accept: application/json" \
  -c "$COOKIES_RT" \
  -D - | grep -i "X-CSRF-Token" | cut -d' ' -f2 | tr -d '\r')
if [ -z "$CSRF_RT" ]; then
  echo "Error: Could not get CSRF token for Rate Table API."
  exit 1
fi

# ----- Step 5: POST $batch to upload rates (using extracted VALIDITY_UUID) -----
# 30 dummy weight breaks (AT->AT lane): 50, 75, 100, 150, 200, ..., 20001 kg
echo "Step 5: Uploading batch rates (30 dummy records)..."
WEIGHTS="50 75 100 150 200 300 400 500 750 1000 1250 1500 1750 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12500 15000 17500 20000 20001"

CHUNK_PARTS=""
ID=1
for W in $WEIGHTS; do
  # Dummy rates: 47.60 (50-100), 53.20 (150-500), 60.00 (750-2000), 70.00 (2500+)
  if [ "$W" -le 100 ]; then RATE=47.60
  elif [ "$W" -le 500 ]; then RATE=53.20
  elif [ "$W" -le 2000 ]; then RATE=60.00
  else RATE=70.00
  fi
  CHUNK_PARTS="${CHUNK_PARTS}
--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: ${ID}

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":${RATE},\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"${W}\"}"
  ID=$((ID + 1))
done

BATCH_BODY="--${BOUNDARY}
Content-Type: multipart/mixed; boundary=changeset_all
${CHUNK_PARTS}

--changeset_all--
--${BOUNDARY}--"

curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST "${RT_BASE}/\$batch" \
  -u "$AUTH" \
  -H "X-CSRF-Token: $CSRF_RT" \
  -H "Accept: multipart/mixed" \
  -H "Content-Type: multipart/mixed;boundary=${BOUNDARY}" \
  -b "$COOKIES_RT" \
  -d "$BATCH_BODY"

echo ""
echo "Done. Agreement ID: $AGREEMENT_ID, Validity UUID: $VALIDITY_UUID"
