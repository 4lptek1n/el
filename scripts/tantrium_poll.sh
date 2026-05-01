#!/usr/bin/env bash
REPO="4lptek1n/Tantrium"
TOKEN="$GITHUB_PERSONAL_ACCESS_TOKEN"
RUN_ID=25197533103

for i in $(seq 1 80); do
  RESULT=$(curl -s \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['status'], r.get('conclusion',''))")
  STATUS=$(echo $RESULT | awk '{print $1}')
  CONCLUSION=$(echo $RESULT | awk '{print $2}')
  echo "[$(date +%H:%M:%S)] attempt $i  status=$STATUS  conclusion=$CONCLUSION"
  if [ "$STATUS" = "completed" ]; then
    echo "DONE conclusion=$CONCLUSION"

    # Download artifacts
    echo "Listing artifacts..."
    ARTIFACTS=$(curl -s \
      -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/artifacts")
    echo "$ARTIFACTS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for a in d.get('artifacts',[]):
    print(f\"  artifact: id={a['id']} name={a['name']} size={a['size_in_bytes']}\")
"
    # Download each artifact
    mkdir -p /tmp/tantrium-results
    echo "$ARTIFACTS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for a in d.get('artifacts',[]):
    print(a['archive_download_url'])
" | while read URL; do
      ANAME=$(echo "$ARTIFACTS" | python3 -c "import json,sys; d=json.load(sys.stdin); [print(a['name']) for a in d.get('artifacts',[]) if a['archive_download_url']==\"$URL\"" 2>/dev/null || echo "artifact")
      echo "Downloading $URL -> /tmp/tantrium-results/${ANAME}.zip"
      curl -sL -H "Authorization: Bearer $TOKEN" "$URL" -o "/tmp/tantrium-results/${ANAME}.zip"
    done

    echo "Files downloaded:"
    ls -lh /tmp/tantrium-results/ 2>/dev/null
    break
  fi
  sleep 30
done
