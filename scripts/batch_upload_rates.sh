#!/bin/bash
# Rate table batch upload - format matches sap_upload_tool.py (no quotes around UUID in POST line)
# Usage: ./batch_upload_rates.sh
# Or: bash batch_upload_rates.sh

set -e
VALIDITY_UUID="0d21a8da-7325-1fd1-82b8-8259a301e94c"
BASE="http://103.152.79.22:8002/sap/opu/odata4/sap/api_transpratetable/srvd_a2x/sap/transportationratetable/0001"

# 1) CSRF and cookies
CSRF_TOKEN=$(curl -s -X GET "${BASE}/" \
  -u "asakshi:Hell@Sakshi122" \
  -H "X-CSRF-Token: Fetch" \
  -H "Accept: application/json" \
  -c cookies.txt \
  -D - | grep -i "X-CSRF-Token" | cut -d' ' -f2 | tr -d '\r')

# 2) Build batch body exactly like Python: POST line has UUID without quotes
BATCH_BODY="--batch_123453
Content-Type: multipart/mixed; boundary=changeset_all

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 1

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":47.60,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"50\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 2

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":47.60,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"75\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 3

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":47.60,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"100\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 4

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":53.20,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"150\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 5

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":53.20,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"AT\",\"TransportationScaleItem03Value\":\"200\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 6

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":68.00,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"BE\",\"TransportationScaleItem03Value\":\"50\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 7

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":68.00,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"BE\",\"TransportationScaleItem03Value\":\"75\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 8

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":68.00,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"BE\",\"TransportationScaleItem03Value\":\"100\"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 9

POST TranspRateTableValidity(TranspRateTableValidityUUID=${VALIDITY_UUID})/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{\"TransportationCalcBase01\":\"SOURCELOC_ZONE\",\"TransportationCalcBase02\":\"DESTLOC_ZONE\",\"TransportationCalcBase03\":\"GROSS_WEIGHT\",\"TransportationRateCurrency\":\"EUR\",\"TransportationRateAmount\":76.00,\"TranspRateTableDimensionIndex\":\"3\",\"TransportationScaleItem01Value\":\"AT\",\"TransportationScaleItem02Value\":\"BE\",\"TransportationScaleItem03Value\":\"150\"}

--changeset_all--
--batch_123453--"

# 3) POST $batch
curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST "${BASE}/\$batch" \
  -u "asakshi:Hell@Sakshi122" \
  -H "X-CSRF-Token: $CSRF_TOKEN" \
  -H "Accept: multipart/mixed" \
  -H "Content-Type: multipart/mixed;boundary=batch_123453" \
  -b cookies.txt \
  -d "$BATCH_BODY"
