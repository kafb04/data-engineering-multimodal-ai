{{ config(materialized='external', location='data/gold/gold_valid_trials.parquet') }}

SELECT
    f.subject_id,
    s.session_role,
    c.class_name,
    COUNT(*) AS n_trials,
    COUNT(*) FILTER (NOT f.has_artifact) AS n_valid
FROM {{ source('silver', 'fact_trial') }} f
JOIN {{ source('silver', 'dim_session') }} s USING (session_id)
JOIN {{ source('silver', 'dim_class') }} c USING (class_id)
GROUP BY ALL
ORDER BY ALL
