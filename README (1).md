# GlobalAI Module 3 - generated files
- `GlobalAI_Module3_Report.pdf` - report covering Requirements 1-5 (submission document)
- `monitoring/` prometheus.yml, recording_rules.yml, alert_rules.yml, alertmanager.yml, grafana_dashboard.json
- `autoscaling/` hpa.yaml (+VPA, PDB), prometheus-adapter.yaml, scheduled-scaling.yaml
- `cost_optimization/` cost_optimizer.py (cache + routing; `python cost_optimizer.py --prompts ../load_testing/test_prompts.csv`), redis-cluster.yaml
- `load_testing/` load_test.jmx (100 -> 1000 users over 5 min), test_prompts.csv
- `analysis/` traffic/cost model, HPA simulation, charts, results JSON, report builder

Validated: YAML/JSON/XML parse; optimizer run on the 26 test prompts.
NOT done (needs your environment): applying manifests to a cluster, running JMeter, Grafana screenshots.
Load-test and cost figures in the report are modelled, not measured. Replace secrets in alertmanager.yml.
