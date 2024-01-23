tpt=0.55
cv=6

python trace/gen_query.py --query trace/query/query_seq512.csv \
    --seq-len 512 --n 1024 --tpt ${tpt} --cv ${cv} \
    --trace trace/query/query_tpt${tpt}_cv${cv}.txt \
    # --gen-query
