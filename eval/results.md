# Evaluation results

30 questions per run. Scores are shares from 0 to 1.

| Run | Tools | Hit@k | Numbers | Text | All correct | Latency (ms) | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rag-off | 1.00 | - | 1.00 | 0.00 | 0.33 | 3047 | 1912 |
| rag-k4-chunk400 | 0.97 | 0.75 | 1.00 | 0.70 | 0.80 | 4244 | 3710 |
| rag-k3-chunk300 | 1.00 | 0.85 | 1.00 | 0.65 | 0.77 | 3926 | 3129 |
| rag-k6-chunk300 | 0.97 | 0.90 | 1.00 | 0.75 | 0.83 | 4354 | 3906 |
| rag-k3-chunk500 | 0.93 | 0.70 | 1.00 | 0.70 | 0.80 | 5519 | 3407 |
| rag-k6-chunk500 | 0.97 | 0.85 | 1.00 | 0.80 | 0.87 | 6902 | 4681 |
| v2-rag-k4-chunk400 | 1.00 | 0.80 | 1.00 | 0.70 | 0.80 | 5204 | 3837 |
| v2-rag-k6-chunk300 | 1.00 | 1.00 | 1.00 | 0.75 | 0.83 | 4966 | 4083 |
