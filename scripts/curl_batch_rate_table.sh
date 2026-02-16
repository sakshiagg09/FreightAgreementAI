#!/bin/bash
# Curl for SAP Transportation Rate Table $batch (same validity & payload as batch_upload_rates.sh)
# Uses validity 0d21a8da-7325-1fd1-82b8-8259a301e94c, AT->AT 50/75/100/150 kg
# Usage: ./curl_batch_rate_table.sh

set -e
BASE="http://103.152.79.22:8002/sap/opu/odata4/sap/api_transpratetable/srvd_a2x/sap/transportationratetable/0001"
VALIDITY_UUID="0d21a8da-7325-1fd1-82b8-8259a301e94c"
BOUNDARY="batch_123453"
CHANGESET="changeset_all"

# 1) Fetch fresh CSRF token and save session cookies
CSRF_TOKEN=$(curl -s -X GET "${BASE}/" \
  -u "asakshi:Hell@Sakshi122" \
  -H "X-CSRF-Token: Fetch" \
  -H "Accept: application/json" \
  -c cookies.txt \
  -D - | grep -i "X-CSRF-Token" | cut -d' ' -f2 | tr -d '\r')

# 2) POST $batch with the token and cookies
curl --location "${BASE}/\$batch" \
  -u "asakshi:Hell@Sakshi122" \
  -H "X-CSRF-Token: ${CSRF_TOKEN}" \
  -H "Accept: multipart/mixed" \
  -H "Content-Type: multipart/mixed;boundary=${BOUNDARY}" \
  -b cookies.txt \
  --data-binary @- << 'BODY'
--batch_123453
Content-Type: multipart/mixed; boundary=changeset_all

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 1

POST TranspRateTableValidity(TranspRateTableValidityUUID=0d21a8da-7325-1fd1-82b8-8259a301e94c)/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{"TransportationCalcBase01":"SOURCELOC_ZONE","TransportationCalcBase02":"DESTLOC_ZONE","TransportationCalcBase03":"GROSS_WEIGHT","TransportationRateCurrency":"EUR","TransportationRateAmount":47.60,"TranspRateTableDimensionIndex":"3","TransportationScaleItem01Value":"AT","TransportationScaleItem02Value":"AT","TransportationScaleItem03Value":"50"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 2

POST TranspRateTableValidity(TranspRateTableValidityUUID=0d21a8da-7325-1fd1-82b8-8259a301e94c)/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{"TransportationCalcBase01":"SOURCELOC_ZONE","TransportationCalcBase02":"DESTLOC_ZONE","TransportationCalcBase03":"GROSS_WEIGHT","TransportationRateCurrency":"EUR","TransportationRateAmount":47.60,"TranspRateTableDimensionIndex":"3","TransportationScaleItem01Value":"AT","TransportationScaleItem02Value":"AT","TransportationScaleItem03Value":"75"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 3

POST TranspRateTableValidity(TranspRateTableValidityUUID=0d21a8da-7325-1fd1-82b8-8259a301e94c)/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{"TransportationCalcBase01":"SOURCELOC_ZONE","TransportationCalcBase02":"DESTLOC_ZONE","TransportationCalcBase03":"GROSS_WEIGHT","TransportationRateCurrency":"EUR","TransportationRateAmount":47.60,"TranspRateTableDimensionIndex":"3","TransportationScaleItem01Value":"AT","TransportationScaleItem02Value":"AT","TransportationScaleItem03Value":"100"}

--changeset_all
Content-Type: application/http
Content-Transfer-Encoding: binary
Content-ID: 4

POST TranspRateTableValidity(TranspRateTableValidityUUID=0d21a8da-7325-1fd1-82b8-8259a301e94c)/_TranspRateTableRate HTTP/1.1
OData-Version: 4.0
OData-MaxVersion: 4.0
Content-Type: application/json
Accept: application/json

{"TransportationCalcBase01":"SOURCELOC_ZONE","TransportationCalcBase02":"DESTLOC_ZONE","TransportationCalcBase03":"GROSS_WEIGHT","TransportationRateCurrency":"EUR","TransportationRateAmount":53.20,"TranspRateTableDimensionIndex":"3","TransportationScaleItem01Value":"AT","TransportationScaleItem02Value":"AT","TransportationScaleItem03Value":"150"}

--changeset_all--
--batch_123453--
BODY
