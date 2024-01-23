cv=6

python trace/gen_real_query.py --query trace/query/query_seq512.csv \
    --seq-len 512 --n 1024 --cv ${cv} \
    --arrival trace/query/arrival-rates.csv \
    --trace trace/query/query_realAr_cv${cv}.txt \
    # --gen-query
