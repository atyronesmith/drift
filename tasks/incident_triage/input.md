```
INCIDENT REPORT — INC-2026-04821
================================================================
Reported:    2026-03-28 14:32 UTC
Reporter:    Sarah Chen (sarah.chen@cascademfg.com)
Account:     Cascade Manufacturing Ltd. (ACCT-4827)
Tier:        Enterprise Gold
SLA:         4-hour response, 24-hour resolution
================================================================

AFFECTED SYSTEM:
  Product:     FlowControl Pro v4.2.1
  Component:   Real-Time Data Pipeline
  Environment: Production (us-west-2)
  Instance:    fc-prod-pipeline-07

INCIDENT DESCRIPTION:
  Starting at approximately 13:45 UTC today, the real-time data
  pipeline on instance fc-prod-pipeline-07 began dropping
  approximately 15-20% of incoming sensor readings from the
  Portland manufacturing floor. The data loss is intermittent —
  readings from temperature sensors (PT100 series) are most
  affected, while pressure sensors appear to be reporting normally.

  Operators noticed the issue when the HMI dashboard showed gaps
  in the temperature trend charts. No alarms were triggered because
  the missing data points were simply absent rather than out-of-range.

  We attempted to restart the pipeline service at 14:10 UTC but
  the data loss resumed after approximately 5 minutes. Memory
  usage on the instance appears elevated (87% vs typical 45-55%)
  but CPU utilization is normal at 12%.

  This is affecting our quality control process — we cannot certify
  the current batch (Batch QC-7814) without complete temperature
  records for the curing process. The batch is valued at
  approximately $340,000 and must be certified by end of shift
  (22:00 UTC) or it will need to be scrapped.

PREVIOUS INCIDENTS:
  - INC-2026-03102 (Feb 15): Similar data loss on fc-prod-pipeline-03,
    resolved by increasing JVM heap from 4GB to 8GB
  - INC-2025-11847 (Nov 22): Pipeline crash on v4.1.0 due to
    malformed sensor packet, fixed in v4.2.0 patch

ERROR LOGS (last 50 lines, filtered):
  14:02:33.441 WARN  [pipeline-ingest] Buffer pool exhausted,
    dropping frame batch (sensors: PT100-A7, PT100-A8, PT100-B3)
  14:02:33.442 WARN  [pipeline-ingest] GC pause 1847ms exceeds
    threshold (200ms)
  14:15:48.102 INFO  [pipeline-ctrl] Service restart initiated
  14:15:52.891 INFO  [pipeline-ctrl] Service started successfully
  14:20:55.334 WARN  [pipeline-ingest] Buffer pool exhausted,
    dropping frame batch (sensors: PT100-A7, PT100-B3, PT100-C1)
  14:20:55.335 WARN  [pipeline-ingest] GC pause 2103ms exceeds
    threshold (200ms)

CONTACT:
  Primary:   Sarah Chen, Process Engineer
             sarah.chen@cascademfg.com | (503) 555-4827 x312
  Escalation: Mike Torres, Plant Manager
             m.torres@cascademfg.com | (503) 555-4827 x101
================================================================
```
