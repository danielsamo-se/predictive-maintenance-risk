set -e

API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "Running smoke tests against ${API_URL}"

echo "Test 1: GET /health"
health_status=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health")
if [ "$health_status" != "200" ]; then
    echo "FAIL: /health returned ${health_status}, expected 200"
    exit 1
fi
echo "OK: /health returned 200"

echo "Test 2: POST /predict"

payload='{"window":['
for i in $(seq 1 50); do
    if [ $i -gt 1 ]; then
        payload="${payload},"
    fi
    row='{"op_setting_1":0.0,"op_setting_2":0.0,"op_setting_3":0.0'
    for j in $(seq 1 21); do
        row="${row},\"sensor_${j}\":0.0"
    done
    row="${row}}"
    payload="${payload}${row}"
done
payload="${payload}]}"

predict_status=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${payload}")

if [ "$predict_status" != "200" ]; then
    echo "FAIL: /predict returned ${predict_status}, expected 200"
    exit 1
fi
echo "OK: /predict returned 200"

echo ""
echo "smoke test ok"
exit 0
