-- Regras do desenho do experimento que os testes genéricos do schema.yml não cobrem.
-- Falha se alguma regra tiver violações; cada linha devolvida é uma regra quebrada.
WITH seq AS (
    SELECT onset_s - LAG(onset_s) OVER (
               PARTITION BY subject_id, session_id, run_id ORDER BY trial_in_run) AS gap_s
    FROM {{ source('silver', 'fact_trial') }}
),
fato AS (SELECT * FROM {{ source('silver', 'fact_trial') }}),
regras AS (
    SELECT 'trial_in_run fora de 0 a 47' AS regra, COUNT(*) AS violacoes
    FROM fato WHERE trial_in_run NOT BETWEEN 0 AND 47
    UNION ALL
    SELECT 'duration_s diferente de 4,0', COUNT(*) FROM fato WHERE duration_s <> 4.0
    UNION ALL
    SELECT 'n_samples diferente de duration_s × 250', COUNT(*) FROM fato WHERE n_samples <> duration_s * 250
    UNION ALL
    SELECT 'onset_s negativo', COUNT(*) FROM fato WHERE onset_s < 0
    UNION ALL
    SELECT 'subject_id fora de A01 a A09', COUNT(*) FROM fato WHERE NOT regexp_full_match(subject_id, 'A0[1-9]')
    UNION ALL
    SELECT 'trial_id diferente da composição das chaves', COUNT(*)
    FROM fato
    WHERE trial_id <> subject_id || '_' || session_id || '_' || run_id || '_t' || lpad(trial_in_run::VARCHAR, 2, '0')
    UNION ALL
    SELECT 'run sem 48 tentativas', COUNT(*)
    FROM (SELECT COUNT(*) AS n FROM fato GROUP BY subject_id, session_id, run_id) WHERE n <> 48
    UNION ALL
    SELECT 'run sem 12 tentativas por classe', COUNT(*)
    FROM (SELECT COUNT(*) AS n FROM fato GROUP BY subject_id, session_id, run_id, class_id) WHERE n <> 12
    UNION ALL
    SELECT 'onset_s fora de ordem no run', COUNT(*) FROM seq WHERE gap_s <= 0
    UNION ALL
    SELECT 'intervalo entre tentativas fora de 7,5 a 8,5 s', COUNT(*) FROM seq WHERE gap_s NOT BETWEEN 7.5 AND 8.5
    UNION ALL
    SELECT 'idade fora de 10 a 100 anos', COUNT(*)
    FROM {{ source('silver', 'dim_subject') }} WHERE age NOT BETWEEN 10 AND 100
)
SELECT * FROM regras WHERE violacoes > 0
